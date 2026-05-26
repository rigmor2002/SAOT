import torch
import torch.nn as nn
import dgl
import copy
import math
import torch.nn.functional as F

from dgl import block_to_graph

try:
    from .losses_ot import OTLoss
    from aug import augmentors as A
    from .subgraph_replay_utils import *
except ImportError:
    from losses_ot import OTLoss
    from aug import augmentors as A
    from subgraph_replay_utils import *

samplers = {'random': random_subgraph_sampler, 'd_samp': degree_based_sampler}

class OTSubgraphReplayModel(nn.Module):

    def __init__(self, model, args):
        super().__init__()
        self.args = args
        device = args.device
        self.device = device

        self.encoder1 = model
        self.encoder2 = copy.deepcopy(model)

        self.encoder1_args = args
        self.encoder2_args = args

        self.pe = getattr(args, 'pe', 0.3)
        self.pf = getattr(args, 'pf', 0.3)
        self.pn = getattr(args, 'pn', 0.0)
        
        self.aug1 = A.Identity()
        self.aug2 = A.Compose([
            A.EdgePerturbation(pe=self.pe),
            A.FeatureMasking(pf=self.pf),
            A.NodeDropping(pn=self.pn)
        ])

        self.ot_loss_fn = OTLoss(args, device=device)

        params = list(self.encoder1.parameters()) + list(self.encoder2.parameters())
        self.optimizer = torch.optim.Adam(
            params, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )

        self.sampler = samplers[args.sgreplay_args['sampler']](args)
        self.current_task = -1
        self.buffer_c_node = []
        self.buffer_all_nodes = []

        self.ot_struct_lambda = args.ot_struct_lambda

        self.task_initialized = False
        self._cached_loader_num_nodes = None

        self.prev_encoder1 = None
        self.prev_encoder2 = None
        self.enable_distill = getattr(args, 'enable_distill', False)
        self.distill_lambda = getattr(args, 'distill_lambda', 10.0)

    def _update_replay_buffer(self, g, train_ids, ids_per_cls, t):

        if t == self.current_task:
            return

        if self.enable_distill and t > 0:

            state1 = self.encoder1.state_dict()
            state2 = self.encoder2.state_dict()

            from Backbones.model_factory import get_model
            from Backbones.utils import NodeLevelDataset

            dataset_temp = NodeLevelDataset(
                self.args.dataset,
                ratio_valid_test=self.args.ratio_valid_test,
                args=self.args
            )

            self.prev_encoder1 = get_model(dataset_temp, self.args).to(self.device)
            self.prev_encoder2 = get_model(dataset_temp, self.args).to(self.device)

            self.prev_encoder1.load_state_dict(state1)
            self.prev_encoder2.load_state_dict(state2)

            self.prev_encoder1.eval()
            self.prev_encoder2.eval()
            for param in self.prev_encoder1.parameters():
                param.requires_grad = False
            for param in self.prev_encoder2.parameters():
                param.requires_grad = False

        self.current_task = t
        self.task_initialized = False

        old_ids = g.ndata["_ID"].cpu()
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]

        c_nodes_sampled, nbs_sampled = self.sampler(
            g,
            self.args.sgreplay_args["c_node_budget"],
            self.args.sgreplay_args["nei_budget"],
            self.encoder1,
            ids_per_cls_train
        )

        self.buffer_c_node.extend(old_ids[c_nodes_sampled])
        self.buffer_all_nodes.append(old_ids[nbs_sampled].tolist())

    def _prepare_training_graph(self, g, train_ids, dataset):
        device = g.device
        extended_ids = list(train_ids)
        merged = False

        base_nodes = g.num_nodes()

        if self.current_task > 0 and len(self.buffer_all_nodes) > 0:
            aux_g, _, _ = dataset.get_graph(node_ids=self.buffer_all_nodes, remove_edges=False)
            if aux_g is not None and aux_g.num_nodes() > 0:
                aux_g = aux_g.to(device)

                for key in g.edata.keys():
                    if key not in aux_g.edata:
                        aux_g.edata[key] = torch.zeros((aux_g.num_edges(), *g.edata[key].shape[1:]), dtype=g.edata[key].dtype, device=device)
                for key in aux_g.edata.keys():
                    if key not in g.edata:
                        g.edata[key] = torch.zeros((g.num_edges(), *aux_g.edata[key].shape[1:]), dtype=aux_g.edata[key].dtype, device=device)


                aux_ids = aux_g.ndata['_ID'].cpu().tolist()
                id_map = {nid: idx for idx, nid in enumerate(aux_ids)}
                for nid in self.buffer_c_node:
                    idx = id_map.get(int(nid))
                    if idx is not None:
                        extended_ids.append(idx + base_nodes)

                g = dgl.batch([g, aux_g])
                merged = True

        return g, extended_ids, merged

    def _augment_and_forward(self, g):
        device = g.device
        features = g.ndata['feat']
        edge_index = torch.stack(g.edges(), dim=0)

        x1, edge_index1, _, _ = self.aug1(features, edge_index)
        x2, edge_index2, _, _ = self.aug2(features, edge_index)

        view1_g = dgl.graph((edge_index1[0], edge_index1[1]), num_nodes=x1.shape[0]).to(device)
        view2_g = dgl.graph((edge_index2[0], edge_index2[1]), num_nodes=x2.shape[0]).to(device)

        _ = self.encoder1(view1_g, x1)
        z1 = self.encoder1.second_last_h
        _ = self.encoder2(view2_g, x2)
        z2 = self.encoder2.second_last_h
        if isinstance(z1, tuple):
            z1 = z1[0]
        if isinstance(z2, tuple):
            z2 = z2[0]

        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        return (x1, z1, edge_index1), (x2, z2, edge_index2)

    def _backward_with_ot_loss(self, view1_data, view2_data):
        from Baselines.losses_ot import rowwise_ce_distill

        x1, z1, edge_index1 = view1_data
        x2, z2, edge_index2 = view2_data

        student_z1 = z1
        student_z2 = z2

        distill_loss_value = 0.0

        if self.enable_distill and self.prev_encoder1 is not None and self.current_task > 0:
            struct_loss, match_loss, P_new, B_new = self.ot_loss_fn.compute_loss(
                view1_data, view2_data, return_plans=True
            )

            with torch.no_grad():
                self.prev_encoder1.eval()
                self.prev_encoder2.eval()

                device = x1.device
                view1_g = dgl.graph((edge_index1[0], edge_index1[1]), num_nodes=x1.shape[0]).to(device)
                view2_g = dgl.graph((edge_index2[0], edge_index2[1]), num_nodes=x2.shape[0]).to(device)

                _ = self.prev_encoder1(view1_g, x1)
                z1_old = self.prev_encoder1.second_last_h
                if isinstance(z1_old, tuple):
                    z1_old = z1_old[0]
                z1_old = F.normalize(z1_old, p=2, dim=1)

                _ = self.prev_encoder2(view2_g, x2)
                z2_old = self.prev_encoder2.second_last_h
                if isinstance(z2_old, tuple):
                    z2_old = z2_old[0]
                z2_old = F.normalize(z2_old, p=2, dim=1)

                _, _, P_old, B_old = self.ot_loss_fn.compute_loss(
                    (x1, z1_old, edge_index1),
                    (x2, z2_old, edge_index2),
                    return_plans=True
                )

            if B_new is not None and B_old is not None:
                if B_new.shape != B_old.shape:
                    min_size = min(B_new.shape[0], B_old.shape[0])
                    B_new = B_new[:min_size, :min_size]
                    B_old = B_old[:min_size, :min_size]

                n_pts = B_new.shape[0]
                scale = math.sqrt(n_pts / 1024.0) if n_pts > 0 else 1.0
                distill_loss = rowwise_ce_distill(B_new, B_old, scale=scale)
                distill_loss_value = distill_loss.item()

            else:
                distill_loss = torch.tensor(0.0, device=x1.device)
                distill_loss_value = 0.0

            ot_loss = self.ot_struct_lambda * struct_loss + match_loss
            total_loss = ot_loss + self.distill_lambda * distill_loss

        else:
            struct_loss, match_loss = self.ot_loss_fn.compute_loss(view1_data, view2_data)
            ot_loss = self.ot_struct_lambda * struct_loss + match_loss
            total_loss = ot_loss
            distill_loss = torch.tensor(0.0, device=x1.device)
            distill_loss_value = 0.0

        if total_loss > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder1.parameters()) + list(self.encoder2.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

        return total_loss, {
            'struct': float(struct_loss.detach()),
            'match': float(match_loss.detach()),
            'distill': float(distill_loss_value)
        }

    def _build_dataloader(self, g, train_ids):

        device = g.device
        if isinstance(train_ids, list):
            train_ids = torch.tensor(train_ids, dtype=torch.long, device=device)
        elif isinstance(train_ids, torch.Tensor):
            train_ids = train_ids.to(device)
        
        return dgl.dataloading.NodeDataLoader(
            g,
            train_ids,
            self.args.nb_sampler,
            batch_size=getattr(self.args, 'batch_size', 1024),
            shuffle=getattr(self.args, 'batch_shuffle', True),
            drop_last=False,
            num_workers=getattr(self.args, 'num_workers', 0)
        )

    def observe(self, g, t, train_ids, ids_per_cls, dataset):

        self.encoder1.train()
        self.encoder2.train()

        self._update_replay_buffer(g, train_ids, ids_per_cls, t)

        g, extended_ids, merged = self._prepare_training_graph(g, train_ids, dataset)
        if merged and not self.task_initialized:
            self.task_initialized = True

        view1_data, view2_data = self._augment_and_forward(g)
        total_loss, loss_breakdown = self._backward_with_ot_loss(view1_data, view2_data)

        return total_loss.item()

    def observe_minibatch(self, g, t, train_ids, ids_per_cls, dataset):

        self.encoder1.train()
        self.encoder2.train()
        device = g.device

        self._update_replay_buffer(g, train_ids, ids_per_cls, t)
        g, extended_ids, merged = self._prepare_training_graph(g, train_ids, dataset)
        if merged and not self.task_initialized:
            self.task_initialized = True

        loader = self._build_dataloader(g, extended_ids)
        last_loss = torch.tensor(0.0, device=device)
        loss_logs = []

        for input_nodes, output_nodes, blocks in loader:
            blocks = [b.to(device) for b in blocks]
            block = blocks[-1]

            src, dst = block.edges()
            num_src_nodes = blocks[0].num_src_nodes()
            block_graph = dgl.graph((src, dst), num_nodes=num_src_nodes, device=device)

            block_graph.ndata['feat'] = blocks[0].srcdata['feat']
            if '_ID' in blocks[0].srcdata:
                block_graph.ndata['_ID'] = blocks[0].srcdata['_ID']
            elif dgl.NID in blocks[0].srcdata:
                block_graph.ndata['_ID'] = blocks[0].srcdata[dgl.NID]
            else:
                block_graph.ndata['_ID'] = torch.arange(num_src_nodes, device=device)

            view1_data, view2_data = self._augment_and_forward(block_graph)
            last_loss, loss_breakdown = self._backward_with_ot_loss(view1_data, view2_data)
            loss_logs.append(loss_breakdown)


        return last_loss.item()

    def forward(self, g, features):

        self.encoder1.eval()
        self.encoder2.eval()
        
        with torch.no_grad():
            _ = self.encoder1(g, features)
            z1 = self.encoder1.second_last_h
            
            _ = self.encoder2(g, features)
            z2 = self.encoder2.second_last_h
            
            if isinstance(z1, tuple):
                z1 = z1[0]
            if isinstance(z2, tuple):
                z2 = z2[0]
            
            return z1 + z2
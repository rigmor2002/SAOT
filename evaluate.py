import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
import json
import ast
from distutils.util import strtobool
from datetime import datetime
from glob import glob

import dgl
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader

from Backbones.model_factory import get_model
from Backbones.utils import NodeLevelDataset
from training.utils import set_seed, mkdir_if_missing
from visualize import show_final_APAF, AP_err, AF_err


class Logger:
    def __init__(self, log_file, mode='a'):
        self.terminal = sys.stdout
        self.log = open(log_file, mode, encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def parse_dict(s):
    if isinstance(s, str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return s
    return s

class FrozenEncoder(nn.Module):
    def __init__(self, encoder1, encoder2):
        super(FrozenEncoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.single_encoder_mode = (encoder1 is encoder2)
        
        for param in self.encoder1.parameters():
            param.requires_grad = False
        if not self.single_encoder_mode:
            for param in self.encoder2.parameters():
                param.requires_grad = False
            
        self.encoder1.eval()
        if not self.single_encoder_mode:
            self.encoder2.eval()
    
    def forward(self, subgraph, return_both=False):
        with torch.no_grad():
            node_features = subgraph.ndata['feat'] if 'feat' in subgraph.ndata else subgraph.ndata['feature']
            _ = self.encoder1(subgraph, node_features)
            z1 = self.encoder1.second_last_h
            
            if self.single_encoder_mode:
                z2 = z1
                z = z1
            else:
                _ = self.encoder2(subgraph, node_features)
                z2 = self.encoder2.second_last_h
                z = z1 + z2
        if return_both:
            return z1, z2
        return z

    def encode_graph(self, subgraph, args):
        if getattr(args, 'eval_minibatch', False):
            return self._encode_minibatch(subgraph, args)
        return self.forward(subgraph)

    def _build_eval_dataloader(self, graph, args):
        fanouts = getattr(args, 'eval_n_nbs_sample', [10, 10])
        sampler = MultiLayerNeighborSampler([int(f) for f in fanouts])
        node_ids = torch.arange(graph.num_nodes(), device=graph.device)
        return NodeDataLoader(
            graph,
            node_ids,
            sampler,
            batch_size=getattr(args, 'eval_batch_size', 4096),
            shuffle=False,
            drop_last=False,
            num_workers=getattr(args, 'eval_num_workers', 0),
        )

    def _encode_minibatch(self, subgraph, args):
        device = next(self.encoder1.parameters()).device
        loader = self._build_eval_dataloader(subgraph, args)
        fused = None
        
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in loader:
                blocks = [b.to(device) for b in blocks]
                if subgraph.device != device:
                    subgraph_feat = subgraph.ndata['feat'].to(device)
                else:
                    subgraph_feat = subgraph.ndata['feat']
                
                feats = subgraph_feat[input_nodes]
                _ = self.encoder1.forward_batch(blocks, feats)
                z1 = self.encoder1.second_last_h
                
                if not self.single_encoder_mode:
                    _ = self.encoder2.forward_batch(blocks, feats)
                    z2 = self.encoder2.second_last_h
                    if isinstance(z2, tuple): z2 = z2[0]
                    z2 = z2[:blocks[-1].num_dst_nodes()]
                else:
                    z2 = z1

                if isinstance(z1, tuple): z1 = z1[0]
                z1 = z1[:blocks[-1].num_dst_nodes()]

                out_idx = output_nodes.to('cpu')
                if fused is None:
                    dim = z1.shape[1]
                    fused = torch.zeros(subgraph.num_nodes(), dim, device='cpu')

                fused[out_idx] = (z1 + z2).cpu() if not self.single_encoder_mode else z1.cpu()
        return fused

def load_pretrained_encoders(model_dir, dataset_name, method_name, task_id, temp_args, temp_dataset):
    model_path = os.path.join(model_dir, dataset_name, method_name, f'task_{task_id}_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for task {task_id}: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')

    encoder1 = get_model(temp_dataset, temp_args)
    encoder2 = get_model(temp_dataset, temp_args)
    encoder1.load_state_dict(checkpoint['encoder1_state_dict'], strict=False)
    encoder2.load_state_dict(checkpoint['encoder2_state_dict'], strict=False)
    frozen_encoder = FrozenEncoder(encoder1, encoder2)

    return frozen_encoder

# ==========================================
# Logistic Regression 
# ==========================================

class LogReg_fit(nn.Module):
    def __init__(self, ft_in, nb_classes, weight_decay, device):
        super(LogReg_fit, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self._optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.01, 
            weight_decay=weight_decay,
        )
        self._loss_fn = nn.CrossEntropyLoss()
        self._num_epochs = 2000 
        self._device = device

        for m in self.modules():
            self.weights_init(m)

        self.to(self._device)
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
    
    def fit(self, train_embs, train_lbls, test_embs, test_lbls):
        self.train()
        epoch_win = 0
        best_acc = 0.0
        
        train_embs = train_embs.to(self._device)
        train_lbls = train_lbls.to(self._device)
        test_embs = test_embs.to(self._device)
        test_lbls = test_lbls.to(self._device)

        for epoch in range(self._num_epochs):
            self._optimizer.zero_grad()
            logits = self(train_embs)
            loss = self._loss_fn(input=logits, target=train_lbls)
            loss.backward()
            self._optimizer.step()

            # Early stopping check
            if (epoch + 1) % 20 == 0: 
                self.eval()
                with torch.no_grad():
                    logits = self(test_embs)
                    preds = torch.argmax(logits, dim=1)
                    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                    
                    if acc >= best_acc:
                        best_acc = acc
                        epoch_win = 0
                    else:
                        epoch_win += 1
                
                self.train()
                
                if epoch_win >= 10: 
                    break
                
        return best_acc.item() if isinstance(best_acc, torch.Tensor) else best_acc
    
    def predict_acc(self, embs, lbls):
        self.eval()
        embs = embs.to(self._device)
        lbls = lbls.to(self._device)
        with torch.no_grad():
            logits = self(embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == lbls).float() / lbls.shape[0]
        return acc.item()


def main():
    parser = argparse.ArgumentParser(description='Incremental Linear Evaluation for OT Representation Learning (Fit Protocol)')

    parser.add_argument("--dataset", type=str, default='Arxiv-CL', help='Dataset name')
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    parser.add_argument('--scenario', type=str, default='task', choices=['task', 'class'],
                        help='Evaluation scenario: task (Task-IL) or class (Class-IL)')

    parser.add_argument('--backbone', type=str, default='GCN', help="Backbone GNN")
    parser.add_argument('--method', type=str, default='ot_replay', help="Method name")
    parser.add_argument('--n_cls_per_task', type=int, default=2, help='Classes per task')
    parser.add_argument('--GCN-args', type=str, default='{"h_dims": [256], "dropout": 0.0}', help='GCN args')
    parser.add_argument('--GAT-args', type=str, default='{"h_dims": [256], "dropout": 0.0, "num_heads": 8, "num_layers": 2, "heads": 8, "out_heads": 1}', help='GAT args')

    parser.add_argument('--ratio_valid_test', nargs='+', default=[0.2, 0.2], help='Validation and test ratios')
    parser.add_argument('--ori_data_path', type=str, default='../store/data', help='Original data path')
    parser.add_argument('--data_path', type=str, default='./data', help='Processed data path')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for LogReg')

    parser.add_argument('--model_dir', type=str, default='./results/models', help='Model directory')
    parser.add_argument('--result_path', type=str, default='./results', help='Result path')

    parser.add_argument('--eval_minibatch', type=strtobool, default=False, help='Minibatch inference')
    parser.add_argument('--eval_batch_size', type=int, default=4096, help='Inference batch size')
    parser.add_argument('--eval_n_nbs_sample', nargs='+', type=int, default=[15, 10], help='Neighbor sampling fanouts')
    parser.add_argument('--eval_num_workers', type=int, default=0)
    
    args = parser.parse_args()
    args.ratio_valid_test = [float(i) for i in args.ratio_valid_test]
    args.GCN_args = parse_dict(args.GCN_args)
    args.GAT_args = parse_dict(args.GAT_args)
    args.eval_n_nbs_sample = [int(i) for i in args.eval_n_nbs_sample]

    if args.backbone == 'GAT':
        num_layers = args.GAT_args.get('num_layers', 2)
        required_blocks = num_layers + 1
        if len(args.eval_n_nbs_sample) != required_blocks:
            last_fanout = args.eval_n_nbs_sample[-1] if args.eval_n_nbs_sample else 10
            if len(args.eval_n_nbs_sample) < required_blocks:
                args.eval_n_nbs_sample = list(args.eval_n_nbs_sample) + [last_fanout] * (required_blocks - len(args.eval_n_nbs_sample))
            else:
                args.eval_n_nbs_sample = args.eval_n_nbs_sample[:required_blocks]

    set_seed(args)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    log_dir = f'{args.result_path}/logs/{args.dataset}/{args.method}/'
    mkdir_if_missing(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{log_dir}eval_{args.scenario}_{timestamp}.log'
    logger = Logger(log_file, mode='w')
    sys.stdout = logger

    dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls

    task_sequence = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls, args.n_cls_per_task)]
    if len(task_sequence[-1]) < args.n_cls_per_task:
        task_sequence.pop()
    n_tasks = len(task_sequence)

    actual_n_tasks = 0
    for task_id in range(n_tasks):
        model_path = os.path.join(args.model_dir, args.dataset, args.method, f'task_{task_id}_model.pt')
        if os.path.exists(model_path):
            actual_n_tasks = task_id + 1
        else:
            break
            
    if actual_n_tasks == 0:
        print("Error: No trained models found.")
        sys.exit(1)
    if actual_n_tasks < n_tasks:
        n_tasks = actual_n_tasks
        task_sequence = task_sequence[:n_tasks]
        
    acc_matrix = np.zeros((n_tasks, n_tasks))

    temp_args = argparse.Namespace()
    temp_args.backbone = args.backbone
    temp_args.GCN_args = args.GCN_args
    temp_args.GAT_args = args.GAT_args
    temp_args.n_cls_per_task = args.n_cls_per_task
    temp_args.d_data = args.d_data
    temp_args.n_cls = args.n_cls

    for current_task_id in range(n_tasks):
        try:
            frozen_encoder = load_pretrained_encoders(
                args.model_dir, args.dataset, args.method, current_task_id, temp_args, dataset
            )
            frozen_encoder = frozen_encoder.to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            break
            
        current_accuracies = []
        
        if args.scenario == 'task':
            
            for t in range(current_task_id + 1):
                task_cls = task_sequence[t]

                subgraph, _, [train_ids, val_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                if not getattr(args, 'eval_minibatch', False):
                    subgraph = subgraph.to(device)
                
                embeddings = frozen_encoder.encode_graph(subgraph, args)

                val_ids = torch.as_tensor(val_ids, dtype=torch.long, device=device)
                test_ids = torch.as_tensor(test_ids, dtype=torch.long, device=device)
                eval_ids = torch.cat([val_ids, test_ids])

                global_labels = subgraph.ndata['label'].squeeze()
                min_cls = min(task_cls)

                train_ids = torch.as_tensor(train_ids, dtype=torch.long, device=device)
                
                train_lbls = global_labels[train_ids] - min_cls
                test_lbls = global_labels[eval_ids] - min_cls
                
                train_embs = embeddings[train_ids]
                test_embs = embeddings[eval_ids]

                feature_dim = embeddings.shape[1]
                num_classes_task = len(task_cls)
                
                logreg = LogReg_fit(feature_dim, num_classes_task, args.weight_decay, device)
                acc = logreg.fit(train_embs, train_lbls, test_embs, test_lbls)
                
                acc_matrix[current_task_id, t] = acc * 100
                current_accuracies.append(acc * 100)
            
        elif args.scenario == 'class':
            all_train_embs_list = []
            all_train_lbls_list = []
            all_test_embs_list = [] 
            all_test_lbls_list = [] 
            
            task_specific_test_data = {} 

            for t in range(current_task_id + 1):
                task_cls = task_sequence[t]
                subgraph, _, [train_ids, val_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
                
                if not getattr(args, 'eval_minibatch', False):
                    subgraph = subgraph.to(device)
                    
                embeddings = frozen_encoder.encode_graph(subgraph, args)
                global_labels = subgraph.ndata['label'].squeeze()

                train_ids = torch.as_tensor(train_ids, dtype=torch.long, device=device)
                val_ids = torch.as_tensor(val_ids, dtype=torch.long, device=device)
                test_ids = torch.as_tensor(test_ids, dtype=torch.long, device=device)
                
                all_train_embs_list.append(embeddings[train_ids].cpu())
                all_train_lbls_list.append(global_labels[train_ids].cpu())

                eval_ids = torch.cat([val_ids, test_ids])
                
                eval_embs = embeddings[eval_ids].cpu()
                eval_lbls = global_labels[eval_ids].cpu()
                
                all_test_embs_list.append(eval_embs)
                all_test_lbls_list.append(eval_lbls)
                
                task_specific_test_data[t] = (eval_embs, eval_lbls)

            full_train_embs = torch.cat(all_train_embs_list, dim=0)
            full_train_lbls = torch.cat(all_train_lbls_list, dim=0)
            full_test_embs = torch.cat(all_test_embs_list, dim=0)
            full_test_lbls = torch.cat(all_test_lbls_list, dim=0)
            
            feature_dim = full_train_embs.shape[1]
            max_cls_seen = max([max(seq) for seq in task_sequence[:current_task_id+1]])
            num_classes_global = max_cls_seen + 1

            logreg = LogReg_fit(feature_dim, num_classes_global, args.weight_decay, device)
            _ = logreg.fit(full_train_embs, full_train_lbls, full_test_embs, full_test_lbls)

            for t in range(current_task_id + 1):
                t_embs, t_lbls = task_specific_test_data[t]
                acc = logreg.predict_acc(t_embs, t_lbls)
                
                acc_matrix[current_task_id, t] = acc * 100
                current_accuracies.append(acc * 100)

        task_str = '|'.join([f'T{i:02d} {acc:.2f}' for i, acc in enumerate(current_accuracies)])
        acc_mean = np.mean(current_accuracies)
        print(f"{task_str}|acc_mean: {acc_mean:.2f}")

    linear_eval_dir = os.path.join(args.result_path, "linear_eval")
    mkdir_if_missing(linear_eval_dir)
    result_filename = f'linear_eval_{args.dataset}_{args.method}_{args.backbone}_{args.scenario}.pkl'
    result_path = os.path.join(linear_eval_dir, result_filename)
    
    with open(result_path, 'wb') as f:
        pickle.dump([acc_matrix], f)

    print("   Final Evaluation Results:")
    
    performance_mean, _, _ = AP_err([acc_matrix])
    AF_mean, _, _ = AF_err([acc_matrix])
    
    if len(performance_mean) > 0:
        print(f"\nAP:  {performance_mean[-1]:.2f}")
        print(f"AF:  {AF_mean:.1f}")
    else:
        print("\nNo results to display.")

    sys.stdout = logger.terminal
    logger.close()

if __name__ == '__main__':
    main()
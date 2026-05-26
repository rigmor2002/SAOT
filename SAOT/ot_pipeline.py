import os
import sys
import pickle
import torch
import numpy as np
from datetime import datetime
from dgl.dataloading import MultiLayerNeighborSampler
from Backbones.model_factory import get_model
from Backbones.utils import NodeLevelDataset
from training.utils import mkdir_if_missing
import copy
import time

from Baselines.ot_subgraph_replay_model import OTSubgraphReplayModel


class Logger:

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()



def pipeline_ot_representation_learning(args, ModelClass=None):

    torch.cuda.set_device(args.gpu)

    log_dir = f'{args.result_path}/logs/{args.dataset}/{args.method}/'
    mkdir_if_missing(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{log_dir}training_{timestamp}.log'

    logger = Logger(log_file)
    sys.stdout = logger


    print("\n" + "="*60)
    print("   OT Representation Learning - Training Phase")
    print("="*60)
    
    dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls

    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls - 1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    
    print(f"\n[Dataset] {args.dataset}")
    print(f"  - Feature dim: {args.d_data}")
    print(f"  - Total classes: {args.n_cls}")
    print(f"  - Classes per task: {args.n_cls_per_task}")
    print(f"  - Total tasks: {args.n_tasks}")
    

    base_model = get_model(dataset, args).cuda(args.gpu)
    life_model = ModelClass(base_model, args)
    

    # ========== Mini-batch Setting ==========
    if args.minibatch:
        args.nb_sampler = MultiLayerNeighborSampler([int(fanout) for fanout in args.n_nbs_sample])
    else:
        args.nb_sampler = None

    
    print(f"\n[Training Started] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=f'cuda:{args.gpu}')
        torch.cuda.empty_cache()
    start_time = time.time()

    for task, task_cls in enumerate(args.task_seq):

        subgraph, ids_per_cls, [train_ids, _, _] = dataset.get_graph(tasks_to_retain=task_cls)
        subgraph = subgraph.to(f'cuda:{args.gpu}')

        for epoch in range(args.epochs):
            if args.minibatch and hasattr(life_model, "observe_minibatch"):
                loss = life_model.observe_minibatch(subgraph, task, train_ids, ids_per_cls, dataset)
            else:
                train_subgraph = subgraph
                MAX_EDGES = 1500000
                if train_subgraph.num_edges() > MAX_EDGES:
                    import dgl
                    src, dst = train_subgraph.edges()
                    keep_idx = torch.randperm(train_subgraph.num_edges())[:MAX_EDGES].to(src.device)

                    train_subgraph_new = dgl.graph((src[keep_idx], dst[keep_idx]), 
                                                   num_nodes=train_subgraph.num_nodes(), 
                                                   device=train_subgraph.device)
                    for k, v in train_subgraph.ndata.items():
                        train_subgraph_new.ndata[k] = v
                    train_subgraph = train_subgraph_new

                loss = life_model.observe(train_subgraph, task, train_ids, ids_per_cls, dataset)


            if (epoch + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        save_dir = f'{args.result_path}/models/{args.dataset}/{args.method}/'
        mkdir_if_missing(save_dir)
        
        save_path = f'{save_dir}task_{task}_model.pt'

        torch.save({
            'encoder1_state_dict': life_model.encoder1.state_dict(),
            'encoder2_state_dict': life_model.encoder2.state_dict(),
            'task': task,
            'task_cls': task_cls,
            'args': vars(args),
            'single_encoder': False
        }, save_path)


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[Training Completed] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    end_time = time.time()
    total_time = end_time - start_time
    print('\n' + '='*60)
    print(f'Total Training & Evaluation Time: {total_time:.2f} seconds')
    print(f'Average Time per Task: {total_time / args.n_tasks:.2f} seconds')
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(device=f'cuda:{args.gpu}') / (1024 ** 2)
        print(f'Max GPU Memory Allocated: {max_mem:.2f} MB')
    print('='*60 + '\n')

    sys.stdout = logger.terminal
    logger.close()
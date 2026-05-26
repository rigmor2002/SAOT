import os
os.environ["PYKEOPS_BUILD_TYPE"] = "cpu"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import sys
import argparse
import torch
import dgl
import ast
import json
import time

from training.utils import set_seed, mkdir_if_missing
from Baselines.ot_subgraph_replay_model import OTSubgraphReplayModel


def parse_dict(s):
    if isinstance(s, str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                print(f"Warning: {s}")
                return s
    return s

from ot_pipeline import pipeline_ot_representation_learning


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimal Transport Subgraph Replay Training')

    parser.add_argument("--dataset", type=str, default='CoraFull-CL', help='Datasets, e.g., Arxiv-CL, CoraFull-CL,Products-CL')
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--backbone', type=str, default='GCN', help="eg.GAT, GCN, GIN, SGC")
    parser.add_argument('--method', type=str, default="ot_subgraph_replay")

    parser.add_argument('--minibatch', type=ast.literal_eval, default=False, help="whether to use the mini-batch training")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--n_nbs_sample', nargs='+', type=int, default=[10, 10], help="number of neighbors to sample per hop")
    parser.add_argument('--n_cls_per_task', default=2, type=int)

    parser.add_argument('--GCN-args', type=str, default='{"h_dims": [512], "dropout": 0.0}')
    parser.add_argument('--GAT-args', type=str,
                        default='{"num_layers": 3, "num_hidden": 128, "num_heads": 4, "heads": 4, "out_heads": 1, "feat_drop": 0.0, "attn_drop": 0.0, "negative_slope": 0.2, "residual": false, "dropout": 0.0}')
    parser.add_argument('--sgreplay_args', type=str, default='{"sampler": "d_samp", "c_node_budget": 100, "nei_budget":[50,50]}')

    parser.add_argument('--ot_struct_lambda', type=float, default=0.5)
    parser.add_argument('--ot_max_points', type=int, default=1024)
    parser.add_argument('--enable_distill', type=ast.literal_eval, default=True,
                        help='whether to enable distillation')
    parser.add_argument('--distill_lambda', type=float, default=1.0)

    parser.add_argument('--ori_data_path', type=str, default='./store/data', help='the root path to raw data')
    parser.add_argument('--data_path', type=str, default='./data', help='the path to processed data (splitted into tasks)')
    parser.add_argument('--result_path', type=str, default='./results', help='the path for saving results')

    parser.add_argument('--ratio_valid_test', nargs='+', default=[0.2, 0.2],type=float, help='ratio of nodes used for valid and test')
    parser.add_argument('--device', type=int, default=0, help='PyTorch ID')
    
    args = parser.parse_args()

    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.gpu)
    set_seed(args)
    mkdir_if_missing(args.data_path)
    mkdir_if_missing(args.result_path)

    
    args.GCN_args = parse_dict(args.GCN_args)
    args.GAT_args = parse_dict(args.GAT_args)
    args.sgreplay_args = parse_dict(args.sgreplay_args)

    ModelClass = OTSubgraphReplayModel

    pipeline_ot_representation_learning(args, ModelClass=ModelClass)

    print("--- Training Completed ---")
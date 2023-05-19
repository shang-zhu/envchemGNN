import torch
from torch_geometric.loader import DataLoader
from dualgraph.mol import smiles2graphwithface
from dualgraph.gnn import GNN, GNN2, GNNwithvn
import os
import numpy as np
import random
from tqdm import tqdm
import argparse
from dualgraph.utils import init_distributed_mode
import pandas as pd
import sys
sys.path.append("dualgraph/")
from dualgraph.dataset import DGPygPCQM4MDataset

def reproduce(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def eval(model, device, loader, args):
    model.eval()
    y_true = []
    features = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            feature = model(batch)

        y_true.append(batch.y.detach().cpu())
        features.append(feature.detach().cpu()) 

    y_true = torch.cat(y_true, dim=0)
    features = torch.cat(features, dim=0)

    return y_true, features

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="GNN baselines on pcqm4m with Pytorch Geometrics")
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gnn", type=str, default="dualgraph2")
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--face-reducer", type=str, default="sum")
    parser.add_argument("--graph-pooling", type=str, default="sum")
    parser.add_argument("--init-face", action="store_true", default=False)
    parser.add_argument("--dropedge-rate", type=float, default=0.1)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--ignore-face", action="store_true", default=False)
    parser.add_argument("--use-global", action="store_true", default=False)
    parser.add_argument("--train-subset", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="", help="tensorboard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--save-test", type=bool, default=True)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0) 
    parser.add_argument("--dropnet", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use-outer", action="store_true", default=False)
    parser.add_argument("--lr-warmup", action="store_true", default=False)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--use-bn", action="store_true", default=False)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--face-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--use-adamw", action="store_true", default=False)
    parser.add_argument("--pooler-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--reload", action="store_true", default=False)
    parser.add_argument('--input_path', type=str, default='data')
    parser.add_argument('--input_csv_name', type=str, default='pcqm4m-v2')

    args = parser.parse_args()
    init_distributed_mode(args)
    print(args)

    reproduce(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device(args.device)

    #clip option
    input_path=args.input_path

    ### automatic dataloading and splitting
    DGPygPCQM4MDataset(root=input_path, file_name=args.input_csv_name, smiles2graph=smiles2graphwithface).process()
    dataset = DGPygPCQM4MDataset(root=input_path, file_name=args.input_csv_name, smiles2graph=smiles2graphwithface)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    task_name=args.input_csv_name.split('/')[-1]

    #creating results folder
    if not os.path.exists('graph_features/'+task_name+'/'):
        os.makedirs('graph_features/'+task_name+'/')
        print("The feature directory is created!")

    
    shared_params = {
            "mlp_hidden_size": args.mlp_hidden_size,
            "mlp_layers": args.mlp_layers,
            "latent_size": args.latent_size,
            "use_layer_norm": args.use_layer_norm,
            "num_message_passing_steps": args.num_layers,
            "global_reducer": args.global_reducer,
            "node_reducer": args.node_reducer,
            "face_reducer": args.face_reducer,
            "dropedge_rate": args.dropedge_rate,
            "dropnode_rate": args.dropnode_rate,
            "ignore_globals": not args.use_global,
            "use_face": not args.ignore_face,
            "dropout": args.dropout,
            "dropnet": args.dropnet,
            "init_face": args.init_face,
            "graph_pooling": args.graph_pooling,
            "use_outer": args.use_outer,
            "residual": args.residual,
            "layernorm_before": args.layernorm_before,
            "parallel": args.parallel,
            "pooler_dropout": args.pooler_dropout,
            "use_bn": args.use_bn,
            "node_attn": args.node_attn,
            "face_attn": args.face_attn,
            "global_attn": args.global_attn,
            "ddi": True
        }
  

    if args.gnn == "dualgraph":
        model = GNN(**shared_params).to(device) 
    elif args.gnn == "dualgraph2":
        model = GNN2(**shared_params).to(device)
    elif args.gnn == "dualgraph2-vn":
        model = GNNwithvn(**shared_params).to(device)
    else:
        raise ValueError("Invalid GNN type")
    
    # print(model)
    
    print("Reload from {}...".format(os.path.join(args.checkpoint_dir)))
    checkpoint = torch.load(os.path.join(args.checkpoint_dir), map_location=device)
    model.load_state_dict(checkpoint)

    y_trues, graph_features = eval(model, device, test_loader, args)

    test_df=pd.DataFrame.from_dict({'y_true':list(y_trues.cpu().detach().numpy())})

    test_df.to_csv('graph_features/'+task_name+'/trues.csv', index=False)
    np.save('graph_features/'+task_name+'/graph_feat.npy', graph_features.cpu().detach().numpy())


if __name__ == "__main__":
    main()

import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from dualgraph.mol import smiles2graphwithface
from dualgraph.gnn import GNN, GNN2, GNNwithvn
import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import random
import io
from dualgraph.utils import WarmCosine, init_distributed_mode
import json
from sklearn.model_selection import KFold

import sys
sys.path.append("dualgraph/")


### importing OGB-LSC
from dualgraph.dataset import DGPygPCQM4MDataset

reg_criterion = torch.nn.MSELoss()

def reproduce(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def get_split_idx(size):
    #BUilding Data Loader
    train_idx_list=random.sample(range(size), int(size*0.8))
    other_idx_list=list(set(list(range(size))) - set(train_idx_list))
    val_idx_list=random.sample(other_idx_list, int(size*0.1))
    test_idx_list=list(set(other_idx_list) - set(val_idx_list))
    return train_idx_list, val_idx_list, test_idx_list

def txt2list(txtfile):
    file = open(txtfile, "r")
    data = file.read()
    set_idx = data.split("\n")
    set_idx=[int(idx) for idx in set_idx if idx!='']
    file.close()
    return list(set(set_idx))

def get_split_idx_from_file(split_folder, kfold_idx):
    train_txt=split_folder+'train_idx_'+str(kfold_idx)+'.txt'
    valid_txt=split_folder+'valid_idx_'+str(kfold_idx)+'.txt'
    test_txt=split_folder+'test_idx_'+str(kfold_idx)+'.txt'

    train_idx=txt2list(train_txt)
    valid_idx=txt2list(valid_txt)
    test_idx=txt2list(test_txt)
    return train_idx, valid_idx, test_idx

def train(model, device, loader, optimizer, scheduler, args):
    model.train()
    loss_accum = 0

    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)

        pred = model(batch).view(-1,)
        # if args.unit_clip:
        #     pred = torch.clamp(pred, min=0, max=1)
        optimizer.zero_grad()
        loss = torch.sqrt(reg_criterion(pred, batch.y))
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_accum += loss.detach().cpu().item()
        if step % args.log_interval == 0:
            pbar.set_description(
                "Iteration loss: {:6.4f} lr: {:.5e}".format(
                    loss_accum / (step + 1), scheduler.get_last_lr()[0]
                )
            )

    return loss_accum / (step + 1)


def eval(model, device, loader, args):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=args.disable_tqdm)):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)
            if args.unit_clip:
                pred = torch.clamp(pred, min=0, max=1)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu()) 

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    return torch.sqrt(reg_criterion(y_true, y_pred))

def test(model, device, loader, args):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=args.disable_tqdm)):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)
            if args.unit_clip:
                pred = torch.clamp(pred, min=0, max=1)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu()) 

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    return y_true, y_pred

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
    parser.add_argument('--split_folder', type=str, default='')
    parser.add_argument('--kfold_idx', type=int, required=True)
    parser.add_argument('--unit_clip', type=bool, default=False)
    parser.add_argument('--result_path', type=str, required=True)

    args = parser.parse_args()
    init_distributed_mode(args)
    print(args)

    save_folder=args.result_path
    if not os.path.exists(save_folder):
        # Create a new directory because it does not exist
        try:
            os.makedirs(save_folder)
        except:
            pass
    reproduce(args.random_seed)

    device = torch.device(args.device)

    #clip option
    input_path=args.input_path

    ### automatic dataloading and splitting
    DGPygPCQM4MDataset(root=input_path, file_name=args.input_csv_name, smiles2graph=smiles2graphwithface).process()
    dataset = DGPygPCQM4MDataset(root=input_path, file_name=args.input_csv_name, smiles2graph=smiles2graphwithface)
    if args.reload:
        task_name=args.input_csv_name.split('/')[-1]+'_pretrained'
    else:
        task_name=args.input_csv_name.split('/')[-1]
    kfold_idx=args.kfold_idx

    if args.split_folder!='':
        train_idx, val_idx, test_idx=get_split_idx_from_file(args.split_folder, args.kfold_idx)
    else:
        train_idx, val_idx, test_idx=get_split_idx(len(dataset))
    

    if args.checkpoint_dir != "":
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.reload:
        shared_params = {
                "mlp_hidden_size": 512,
                "mlp_layers": 2,
                "latent_size": 256,
                "use_layer_norm": False,
                "num_message_passing_steps": 12,
                "global_reducer": "sum",
                "node_reducer": "sum",
                "face_reducer": "sum",
                "dropedge_rate": 0.1,
                "dropnode_rate": 0.1,
                "ignore_globals": True,
                "use_face": True,
                "dropout": 0.0,
                "dropnet": 0.0,
                "init_face": True,
                "graph_pooling": "sum",
                "use_outer": False,
                "residual": False,
                "layernorm_before": False,
                "parallel": False,
                "pooler_dropout": 0.0,
                "use_bn": True,
                "node_attn": True,
                "face_attn": False,
                "global_attn": False,
            }
    else:
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
            "global_attn": args.global_attn
        }
    
    if len(val_idx)==0:
        name_appendix=-1
        k=5
        kf = KFold(n_splits= k, shuffle=True, random_state=1)
        for train_index, val_index in kf.split(train_idx):
            name_appendix+=1
            train_loader = DataLoader(dataset[train_index], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            valid_loader = DataLoader(dataset[val_index], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
            test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

            if args.gnn == "dualgraph":
                model = GNN(**shared_params).to(device) 
            elif args.gnn == "dualgraph2":
                model = GNN2(**shared_params).to(device)
            elif args.gnn == "dualgraph2-vn":
                model = GNNwithvn(**shared_params).to(device)
            else:
                raise ValueError("Invalid GNN type")

            model_without_ddp = model
            args.disable_tqdm = False
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
                model_without_ddp = model.module

                args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
                # args.enable_tb = False if args.rank != 0 else args.enable_tb
                args.disable_tqdm = args.rank != 0

            num_params = sum(p.numel() for p in model_without_ddp.parameters())
            print(f"#Params: {num_params}")

            if args.use_adamw:
                optimizer = optim.AdamW(
                    model_without_ddp.parameters(),
                    lr=args.lr,
                    betas=(0.9, args.beta2),
                    weight_decay=args.weight_decay,
                )
            else:
                optimizer = optim.Adam(
                    model_without_ddp.parameters(),
                    lr=args.lr,
                    betas=(0.9, args.beta2),
                    weight_decay=args.weight_decay,
                )

            if args.log_dir != "":
                writer = SummaryWriter(log_dir=args.log_dir)

            best_valid_mae = 1000
            best_test_mae = 1000

            if args.train_subset:
                scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
                args.epochs = 1000
            else:
                if not args.lr_warmup:
                    scheduler = CosineAnnealingLR(optimizer, len(train_loader) * args.period, 1e-7)
                else:
                    lrscheduler = WarmCosine(tmax=len(train_loader) * args.period)
                    scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))
            _start_epoch = 0
            if args.reload:
                print("Reload from {}...".format(os.path.join(args.checkpoint_dir, "checkpoint.pt")))
                checkpoint = torch.load(os.path.join(args.checkpoint_dir, "checkpoint.pt"))
                _start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                # best_valid_mae = checkpoint["best_val_mae"]

            for epoch in range(_start_epoch + 1, args.epochs + 1):
                print("=====Epoch {}".format(epoch))
                print("Training...")
                train_mae = train(model, device, train_loader, optimizer, scheduler, args)

                print("Evaluating...")
                valid_mae = eval(model, device, valid_loader, args)
                test_mae = eval(model, device, test_loader, args)

                print({"Train": train_mae, \
                    "Validation": valid_mae.detach().cpu().item(), \
                        "Test": test_mae.detach().cpu().item()})

                if args.log_dir != "":
                    writer.add_scalar("test/mae", test_mae, epoch)
                    writer.add_scalar("valid/mae", valid_mae, epoch)
                    writer.add_scalar("train/mae", train_mae, epoch)

                if valid_mae < best_valid_mae:
                    best_valid_mae = valid_mae
                    best_test_mae = test_mae

                    if args.save_test: 
                        torch.save(model.state_dict(), save_folder+'/model_k'+str(kfold_idx)+'_n'\
                            +str(name_appendix)+'.pt')
                        print("Predicting on test data...")
                        y_true, y_pred = test(model, device, test_loader, args)
                        if name_appendix==0:
                            test_df=pd.DataFrame.from_dict({'idx':test_idx,'y_true':list(y_true.cpu().detach().numpy()),\
                                'y_pred'+str(name_appendix):list(y_pred.cpu().detach().numpy())})
                        else:
                            test_df=pd.read_csv(save_folder+'/preds_'+str(kfold_idx)+'.csv')
                            test_df['y_pred'+str(name_appendix)]=list(y_pred.cpu().detach().numpy())
                        test_df.to_csv(save_folder+'/preds_'+str(kfold_idx)+'.csv', index=False)

                scheduler.step()
                print(f"Best validation MAE so far: {best_valid_mae.detach().cpu().item()},\
                    with test MAE: {best_test_mae.detach().cpu().item()}")

            if args.log_dir != "":
                writer.close()
            if args.distributed:
                torch.distributed.destroy_process_group()
            
            # with open(save_folder+'result.txt', 'a+') as f:
            #     f.write(f'data({args.random_seed})-best valid/test MAE,{args.input_csv_name}, {args.random_seed},\
            #         {best_valid_mae.detach().cpu().item()}, {best_test_mae.detach().cpu().item()}\n')
    
    else:
        train_loader = DataLoader(dataset[train_idx], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        valid_loader = DataLoader(dataset[val_idx], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(dataset[test_idx], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)


        if args.gnn == "dualgraph":
            model = GNN(**shared_params).to(device) 
        elif args.gnn == "dualgraph2":
            model = GNN2(**shared_params).to(device)
        elif args.gnn == "dualgraph2-vn":
            model = GNNwithvn(**shared_params).to(device)
        else:
            raise ValueError("Invalid GNN type")

        model_without_ddp = model
        args.disable_tqdm = False
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            model_without_ddp = model.module

            args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
            # args.enable_tb = False if args.rank != 0 else args.enable_tb
            args.disable_tqdm = args.rank != 0

        num_params = sum(p.numel() for p in model_without_ddp.parameters())
        print(f"#Params: {num_params}")

        if args.use_adamw:
            optimizer = optim.AdamW(
                model_without_ddp.parameters(),
                lr=args.lr,
                betas=(0.9, args.beta2),
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model_without_ddp.parameters(),
                lr=args.lr,
                betas=(0.9, args.beta2),
                weight_decay=args.weight_decay,
            )

        if args.log_dir != "":
            writer = SummaryWriter(log_dir=args.log_dir)

        best_valid_mae = 1000
        best_test_mae = 1000

        if args.train_subset:
            scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
            args.epochs = 1000
        else:
            if not args.lr_warmup:
                scheduler = CosineAnnealingLR(optimizer, len(train_loader) * args.period, 1e-7)
            else:
                lrscheduler = WarmCosine(tmax=len(train_loader) * args.period)
                scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))
        _start_epoch = 0
        if args.reload:
            print("Reload from {}...".format(os.path.join(args.checkpoint_dir, "checkpoint.pt")))
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, "checkpoint.pt"))
            _start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            # best_valid_mae = checkpoint["best_val_mae"]

        for epoch in range(_start_epoch + 1, args.epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            train_mae = train(model, device, train_loader, optimizer, scheduler, args)

            print("Evaluating...")
            valid_mae = eval(model, device, valid_loader, args)
            test_mae = eval(model, device, test_loader, args)

            print({"Train": train_mae, \
                "Validation": valid_mae.detach().cpu().item(), \
                    "Test": test_mae.detach().cpu().item()})

            if args.log_dir != "":
                writer.add_scalar("test/mae", test_mae, epoch)
                writer.add_scalar("valid/mae", valid_mae, epoch)
                writer.add_scalar("train/mae", train_mae, epoch)

            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                best_test_mae = test_mae

                if args.save_test: 
                    torch.save(model.state_dict(), save_folder+'/model.pt')
                    print("Predicting on test data...")
                    y_true, y_pred = test(model, device, test_loader, args)
                    test_df=pd.DataFrame.from_dict({'idx':test_idx,'y_true':list(y_true.cpu().detach().numpy()),\
                        'y_pred':list(y_pred.cpu().detach().numpy())})
                    test_df.to_csv(save_folder+'/preds_'+str(kfold_idx)+'.csv', index=False)

            scheduler.step()
            print(f"Best validation MAE so far: {best_valid_mae.detach().cpu().item()},\
                with test MAE: {best_test_mae.detach().cpu().item()}")

        if args.log_dir != "":
            writer.close()
        if args.distributed:
            torch.distributed.destroy_process_group()
        
        # with open(save_folder+'/result.txt', 'a+') as f:
        #     f.write(f'data({args.random_seed})-best valid/test MAE,{args.input_csv_name}, {args.random_seed},\
        #         {best_valid_mae.detach().cpu().item()}, {best_test_mae.detach().cpu().item()}\n')


if __name__ == "__main__":
    main()

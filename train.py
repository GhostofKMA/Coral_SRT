import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.dvt import DVT
from model.dataset import CustomDataset

def get_args():
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument('--model_name', type=str,required=True,choices=["SAM","SAM_2","DINO_S_16","DINOv2_B_14","DINOv2_S_14","CoralSCOP"], help='Name of the model to use')
    parser.add_argument('--mode', type=str, choices=['average','median'], default='median', help='Method used for target features')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW')
    return parser.parse_args()

def train():
    device = "cuda"
    args = get_args()
    model_name = args.model_name
    mode = args.mode
    if model_name == "SAM":
        raw_feat_dir = "D:/Coral_SRT/data/raw_features/SAM"
        if mode == 'average':
            target_feat_dir = "D:/Coral_SRT/data/target_features/SAM/mean"
        else:
            target_feat_dir = "D:/Coral_SRT/data/target_features/SAM/median"
    elif model_name == "SAM_2":
        raw_feat_dir = "D:/Coral_SRT/data/raw_features/SAM2"
        if mode == 'average':
            target_feat_dir = "D:/Coral_SRT/data/target_features/SAM2/mean"
        else:
            target_feat_dir = "D:/Coral_SRT/data/target_features/SAM2/median"
    elif model_name == "DINO_S_16":
        raw_feat_dir = "D:/Coral_SRT/data/raw_features/DINO_Small_16"
        if mode == 'average':
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINO_Small_16/mean"
        else:
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINO_Small_16/median"
    elif model_name == "DINOv2_B_14":
        raw_feat_dir = "D:/Coral_SRT/data/raw_features/DINOv2_Base_14"
        if mode == 'average':
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINOv2_Base_14/mean"
        else:
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINOv2_Base_14/median"
    elif model_name == "DINOv2_S_14":
        raw_feat_dir = "D:/Coral_SRT/data/raw_features/DINOv2_Small_14"
        if mode == 'average':
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINOv2_Small_14/mean"
        else:
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINOv2_Small_14/median"
    elif model_name == "CoralSCOP":
        raw_feat_dir = "D:/Coral_SRT/data/raw_features/CoralSCOP"
        if mode == 'average':
            target_feat_dir = "D:/Coral_SRT/data/target_features/CoralSCOP/mean"
        else:
            target_feat_dir = "D:/Coral_SRT/data/target_features/CoralSCOP/median"
    save_path_dir = "D:/Coral_SRT/save_models"
    dataset = CustomDataset(raw_feat_dir, target_feat_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    sample_input, _ = dataset[0]
    feature_dim = sample_input.shape[0]
    model = DVT(feature_dim=feature_dim, num_transformer_blocks=4).to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (raw_batch, target_batch) in enumerate(pbar):
            raw_batch = raw_batch.to(device)
            target_batch = target_batch.to(device)
            output_batch = model(raw_batch)
            mse_loss = F.mse_loss(output_batch, target_batch) 
            pred_flat = output_batch.permute(0,2,3,1).flatten(0,2)
            target_flat = target_batch.permute(0,2,3,1).flatten(0,2)
            cos_loss = 1 - F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
            loss = mse_loss + cos_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item() 
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.6f}")

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            ckpt_name = f"{args.model_name}_{args.mode}_ep{epoch+1}.pth"
            torch.save(model.state_dict(), os.path.join(save_path_dir, ckpt_name))
            print(f"Saved checkpoint: {ckpt_name}")

if __name__ == "__main__":
    train()
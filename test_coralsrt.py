import argparse
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import os
from PIL import Image
from tqdm import tqdm

from model.dvt import DVT

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate CoralSRT (Metrics Only)")
    parser.add_argument('--model_name', type=str, required=True, choices=["SAM","SAM_2","DINO_S_16","DINOv2_B_14","DINOv2_S_14","CoralSCOP"], help='Name of the model to use')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test images')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--num_points', type=int, required=True, choices=[5,10,20,50,100], help='Number of sparse labels per image')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for computation')
    return parser.parse_args()

def sample_sparse_labels(label_mask, total_num_points):
    ys, xs = np.where(label_mask > 0)
    num_valid_pixels = len(xs)
    if num_valid_pixels == 0:
        return torch.tensor([]), torch.tensor([])
    if num_valid_pixels < total_num_points:
        chosen_indices = np.arange(num_valid_pixels)
    else:
        chosen_indices = np.random.choice(num_valid_pixels, total_num_points, replace=False)
    
    sampled_coords = []
    sampled_labels = []
    for idx in chosen_indices:
        y, x = ys[idx], xs[idx]
        sampled_coords.append((y, x))
        sampled_labels.append(label_mask[y, x])
    return torch.tensor(sampled_coords), torch.tensor(sampled_labels)

def perform_knn(feature_map, support_coords, support_labels):
    C, H, W = feature_map.shape

    query_feats = feature_map.reshape(C, -1).T.cpu().numpy() 
    query_feats = np.ascontiguousarray(query_feats, dtype=np.float32)

    flat_feats = feature_map.reshape(C, -1)
    flat_indices = support_coords[:, 0] * W + support_coords[:, 1]
    
    support_feats = flat_feats[:, flat_indices].T.cpu().numpy()  
    support_feats = np.ascontiguousarray(support_feats, dtype=np.float32)
    
    support_labels = support_labels.cpu().numpy()

    faiss.normalize_L2(support_feats)
    faiss.normalize_L2(query_feats)

    index = faiss.IndexFlatIP(C)
    index.add(support_feats)
    
    D, I = index.search(query_feats, k=1)
    
    predicted_labels = support_labels[I.flatten()]  
    pred_mask = torch.from_numpy(predicted_labels).view(H, W)
    return pred_mask

def compute_metrics(pred, gt):
    ids = torch.unique(torch.cat([pred, gt]))
    ious = []
    accs = []
    for uid in ids:
        pred_mask = (pred == uid)
        gt_mask = (gt == uid)
        intersection = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()
        pred_count = pred_mask.sum().item()
        if union > 0:
            ious.append(intersection/union)
        if pred_count > 0:
            accs.append(intersection/pred_count)
        else:
            if gt_mask.sum().item() > 0: 
                accs.append(0.0)

    miou = torch.mean(torch.tensor(ious)).item() if len(ious) > 0 else 0.0
    mpa = torch.mean(torch.tensor(accs)).item() if len(accs) > 0 else 0.0
    return miou, mpa

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    args = get_args()
    device = torch.device(args.device)
    raw_feat_dir = args.test_path
    mask_dir = "D:/Coral_SRT/data/test/masks" 
    feat_files = [f for f in os.listdir(raw_feat_dir) if f.endswith('.npy')]

    sample_feat = np.load(os.path.join(raw_feat_dir, feat_files[0]))
    feature_dim = sample_feat.shape[0]
    

    model = DVT(feature_dim=feature_dim, num_transformer_blocks=4).to(device)
    
    if os.path.exists(args.checkpoint_path):
        print(f"🔄 Loading checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and 'denoiser' in checkpoint:
             model.load_state_dict(checkpoint['denoiser'], strict=False)
        else:
            model.load_state_dict(checkpoint)
       
    model.to(device)
    model.eval()
    
    all_ious = []
    all_accs = []

    for feat_name in tqdm(feat_files):
        base_name = os.path.splitext(feat_name)[0]
        feat_path = os.path.join(raw_feat_dir, feat_name)
        mask_path = os.path.join(mask_dir, base_name + ".png")
        
        if not os.path.exists(mask_path):
            continue
        
        features = np.load(feat_path)
        raw_feat_tensor = torch.from_numpy(features).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            rec_feat_tensor = model(raw_feat_tensor)

        gt_mask_pil = Image.open(mask_path)
        W_orig, H_orig = gt_mask_pil.size
        gt_mask_np = np.array(gt_mask_pil)

        rec_feat_upsampled = F.interpolate(
            rec_feat_tensor, 
            size=(H_orig, W_orig), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)

        coords, labels = sample_sparse_labels(gt_mask_np, args.num_points)
        if len(coords) == 0:
            continue 
            
        pred_mask_orig = perform_knn(rec_feat_upsampled, coords, labels)
        
        gt_mask_tensor = torch.from_numpy(gt_mask_np).long()
        miou, mpa = compute_metrics(pred_mask_orig, gt_mask_tensor)
        
        all_ious.append(miou)
        all_accs.append(mpa)

    avg_miou = np.mean(all_ious) * 100 if len(all_ious) > 0 else 0
    avg_mpa = np.mean(all_accs) * 100 if len(all_accs) > 0 else 0

    print("\n" + "="*40)
    print(f"RESULT FOR {args.model_name} with {args.num_points} points")
    print(f"Mean IoU: {avg_miou:.2f}")
    print(f"Mean PA : {avg_mpa:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Fast Baseline Evaluation (Metrics Only)")
    parser.add_argument('--feature_dir', type=str, required=True, help='Path to .npy features')
    parser.add_argument('--mask_dir', type=str, default="/home/duy-anh/Coral_SRT/data/test/masks")
    parser.add_argument('--num_points', type=int, default=5, choices=[5,10,20,50,100])
    parser.add_argument('--test_size', type=int, default=1024, help='Evaluation resolution')
    return parser.parse_args()

def sample_points(label_mask, total_num_points):
    unique_classes = np.unique(label_mask)
    sampled_coords = []
    sampled_labels = []
    
    for cls in unique_classes:
        ys, xs = np.where(label_mask == cls)
        if len(xs) > 0:
            idx = np.random.choice(len(xs))
            sampled_coords.append((ys[idx], xs[idx]))
            sampled_labels.append(cls)
            
    current = len(sampled_coords)
    if current < total_num_points:
        rem = total_num_points - current
        ys_all, xs_all = np.where(label_mask > -1)
        if len(xs_all) > 0:
            idx_list = np.random.choice(len(xs_all), rem, replace=False)
            sampled_coords.extend(zip(ys_all[idx_list], xs_all[idx_list]))
            sampled_labels.extend(label_mask[ys_all[idx_list], xs_all[idx_list]])
                
    return torch.tensor(sampled_coords), torch.tensor(sampled_labels)

def perform_knn(feature_map, support_coords, support_labels):
    device = feature_map.device
    C, H, W = feature_map.shape
    
    query_feats = feature_map.view(C, -1).permute(1, 0) 
    query_feats = F.normalize(query_feats, p=2, dim=1)
    
    flat_indices = support_coords[:, 0] * W + support_coords[:, 1]
    flat_feats = feature_map.view(C, -1)
    support_feats = flat_feats[:, flat_indices.to(device)].permute(1, 0)
    support_feats = F.normalize(support_feats, p=2, dim=1)

    sim_matrix = torch.matmul(query_feats, support_feats.T)
    _, best_indices = torch.max(sim_matrix, dim=1)
    
    pred_labels = support_labels.to(device)[best_indices]
    return pred_labels.view(H, W)

def fast_compute_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    ids = torch.unique(torch.cat([pred, gt]))
    
    ious, accs = [], []
    for uid in ids:
        p_mask = (pred == uid)
        g_mask = (gt == uid)
        
        inter = (p_mask & g_mask).sum().float()
        union = (p_mask | g_mask).sum().float()
        pred_count = p_mask.sum().float()
        
        if union > 0: ious.append(inter / union)
        if pred_count > 0: accs.append(inter / pred_count)
        elif g_mask.sum() > 0: accs.append(torch.tensor(0.0, device=pred.device))
            
    miou = torch.stack(ious).mean().item() if ious else 0.0
    mpa = torch.stack(accs).mean().item() if accs else 0.0
    return miou, mpa

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    args = get_args()
    
    files = [f for f in os.listdir(args.feature_dir) if f.endswith('.npy')]
    ious, accs = [], []
    
    with torch.no_grad():
        for f in tqdm(files):
            base = os.path.splitext(f)[0]
            mask_path = os.path.join(args.mask_dir, base + ".png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(args.mask_dir, base + ".jpg")
                if not os.path.exists(mask_path): continue
            
            feat_np = np.load(os.path.join(args.feature_dir, f))
            raw_feat = torch.from_numpy(feat_np).float().to(device).unsqueeze(0) 
  
            gt_pil = Image.open(mask_path)
            W_orig, H_orig = gt_pil.size
            scale = args.test_size / max(W_orig, H_orig)
            new_W, new_H = int(W_orig * scale), int(H_orig * scale)
            
            gt_pil_resized = gt_pil.resize((new_W, new_H), Image.NEAREST)
            gt_np = np.array(gt_pil_resized)
            
            rec_feat = F.interpolate(raw_feat, size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
            
            coords, labels = sample_points(gt_np, args.num_points)
            if len(coords) == 0: continue
            
            pred_mask = perform_knn(rec_feat, coords, labels)
            
            gt_tensor = torch.from_numpy(gt_np).long().to(device)
            miou, mpa = fast_compute_metrics(pred_mask, gt_tensor)
            
            ious.append(miou)
            accs.append(mpa)
            
    print(f"\nRESULT ({args.num_points} points):")
    print(f"Mean IoU: {np.mean(ious)*100:.2f}")
    print(f"Mean PA : {np.mean(accs)*100:.2f}")

if __name__ == "__main__":
    main()
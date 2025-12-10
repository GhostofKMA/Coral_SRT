import argparse
import numpy as np
import torch
import faiss
import os
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Bảng màu (giữ nguyên để output ảnh đẹp)
colors = [[167, 18, 159], [180, 27, 92], [104, 139, 233], [49, 198, 135], [98, 207, 26], [118, 208, 133],
          [158, 118, 90], [12, 72, 166], [69, 79, 238], [81, 195, 49], [221, 236, 52], [160, 200, 222],
          [255, 63, 216], [16, 94, 7], [226, 47, 64], [183, 108, 5],
          [55, 252, 193], [147, 154, 196], [233, 78, 165], [108, 25, 95], [184, 221, 46], [54, 205, 145],
          [14, 101, 210], [199, 232, 230], [66, 10, 103], [161, 228, 59], [108, 2, 104], [13, 49, 127],
          [186, 99, 38], [97, 140, 246], [44, 114, 202], [36, 31, 118], [146, 77, 143],
          [188, 100, 14], [131, 69, 63]]

def get_args():
    parser = argparse.ArgumentParser(description="Test Baseline (Raw Features + k-NN)")
    parser.add_argument('--feature_dir', type=str, required=True, help='Path to the .npy feature files')
    parser.add_argument('--mask_dir', type=str, default="D:/Coral_SRT/data/test/masks", help='Path to ground truth masks')
    parser.add_argument('--result_dir', type=str, default="D:/Coral_SRT/results/baseline", help='Folder to save result images')
    parser.add_argument('--num_points', type=int, default=5, choices=[5,10,20,50,100], help='Number of sparse labels per image')
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
    # feature_map shape: (C, H, W)
    C, H, W = feature_map.shape

    query_feats = feature_map.reshape(C, -1).T.cpu().numpy()  # (H*W, C)
    query_feats = np.ascontiguousarray(query_feats, dtype=np.float32)

    flat_feats = feature_map.reshape(C, -1)
    flat_indices = support_coords[:, 0] * W + support_coords[:, 1]
    
    support_feats = flat_feats[:, flat_indices].T.cpu().numpy()  # (num_support, C)
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

def save_colored_mask(mask_array, save_path):
    mask_img = Image.fromarray(mask_array.astype(np.uint8), mode='P')
    palette = []
    for color in colors:
        palette.extend(color)
    if len(palette) < 768:
        palette.extend([0] * (768 - len(palette)))
    mask_img.putpalette(palette)
    mask_img.save(save_path)

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
        if pred_count >0:
            accs.append(intersection/pred_count)
        else:
            if gt_mask.sum().item() > 0: 
                accs.append(0.0)

    miou = torch.mean(torch.tensor(ious)).item() if len(ious) > 0 else 0.0
    mpa = torch.mean(torch.tensor(accs)).item() if len(accs) > 0 else 0.0
    return miou, mpa

def main():
    device = 'cuda'
    np.random.seed(42)
    torch.manual_seed(42)
    args = get_args()
    
    save_dir = os.path.join(args.result_dir, f"{args.num_points}_points")
    os.makedirs(save_dir, exist_ok=True)
    
    feat_files = [f for f in os.listdir(args.feature_dir) if f.endswith('.npy')]
    

    all_ious = []
    all_accs = []

    for feat_name in tqdm(feat_files):
        base_name = os.path.splitext(feat_name)[0]
        feat_path = os.path.join(args.feature_dir, feat_name)
        mask_path = os.path.join(args.mask_dir, base_name + ".png") 
        if not os.path.exists(mask_path):
            continue
        features = np.load(feat_path)

        raw_feat_tensor = torch.from_numpy(features).float().unsqueeze(0).float().to(device)
        gt_mask_pil = Image.open(mask_path)
        W_orig, H_orig = gt_mask_pil.size
        gt_mask_np = np.array(gt_mask_pil)
        rec_feat_upsampled = F.interpolate(raw_feat_tensor, size=(H_orig,W_orig),mode='bilinear',align_corners=False).squeeze(0)    

        coords, labels = sample_sparse_labels(gt_mask_np, args.num_points)
        
        if len(coords) == 0:
            continue

        pred_mask_tensor = perform_knn(rec_feat_upsampled, coords, labels)
        gt_mask_tensor = torch.from_numpy(gt_mask_np).long().to(device)
        miou, mpa = compute_metrics(pred_mask_tensor, gt_mask_tensor)
        all_ious.append(miou)
        all_accs.append(mpa)


        save_path = os.path.join(save_dir, base_name + ".png")
        save_colored_mask(pred_mask_tensor.cpu().numpy(), save_path)

    avg_miou = np.mean(all_ious) * 100 if len(all_ious) > 0 else 0
    avg_mpa = np.mean(all_accs) * 100 if len(all_accs) > 0 else 0

    print("\n" + "="*40)
    print(f"📊 BASELINE RESULTS (k-NN, No Training)")
    print(f"Feature Source: {os.path.basename(args.feature_dir)}")
    print(f"Num Points:     {args.num_points}")
    print(f"Mean IoU:       {avg_miou:.2f}")
    print(f"Mean PA :       {avg_mpa:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
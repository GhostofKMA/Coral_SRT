import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from model.dvt import DVT # Class model của cậu
from model.dataset import CustomDataset # Để lấy logic load nếu cần (hoặc load tay)

# --- BẢNG MÀU (UCSD Palette) ---
UCSD_PALETTE = np.array([
    [0, 0, 0], # Background
    [167, 18, 159], [180, 27, 92], [104, 139, 233], [49, 198, 135], [98, 207, 26],
    [118, 208, 133], [158, 118, 90], [12, 72, 166], [69, 79, 238], [81, 195, 49],
    [221, 236, 52], [160, 200, 222], [255, 63, 216], [16, 94, 7], [226, 47, 64],
    [183, 108, 5], [55, 252, 193], [147, 154, 196], [233, 78, 165], [108, 25, 95],
    [184, 221, 46], [54, 205, 145], [14, 101, 210], [199, 232, 230], [66, 10, 103],
    [161, 228, 59], [108, 2, 104], [13, 49, 127], [186, 99, 38], [97, 140, 246],
    [44, 114, 202], [36, 31, 118], [146, 77, 143], [188, 100, 14], [131, 69, 63]
])

def get_args():
    parser = argparse.ArgumentParser(description="Visualize CoralSRT Results")
    parser.add_argument('--image_path', type=str, required=True, help='Path to original image (.jpg)')
    parser.add_argument('--feature_path', type=str, required=True, help='Path to raw feature (.npy)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--save_path', type=str, default='vis_result.png', help='Output image path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def compute_pca(feature_map):
    """
    Biến đổi Feature (C, H, W) thành Ảnh RGB (H, W, 3) dùng PCA.
    """
    C, H, W = feature_map.shape
    # Flatten: (H*W, C)
    flat_feat = feature_map.reshape(C, -1).T
    
    # Chuẩn hóa (Normalize) trước khi PCA để màu đẹp hơn
    flat_feat = (flat_feat - flat_feat.mean(axis=0)) / (flat_feat.std(axis=0) + 1e-5)

    # Dùng PCA giảm xuống 3 chiều
    pca = PCA(n_components=3)
    pca_feat = pca.fit_transform(flat_feat)
    
    # Reshape lại thành ảnh (H, W, 3)
    pca_img = pca_feat.reshape(H, W, 3)
    
    # Min-Max Normalize về [0, 1] để hiển thị
    pca_img = (pca_img - pca_img.min()) / (pca_img.max() - pca_img.min())
    return pca_img

def colorize_mask(mask):
    """
    Tô màu cho Mask ID dựa trên bảng màu UCSD.
    """
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    for uid in np.unique(mask):
        if uid >= len(UCSD_PALETTE): continue # Tránh lỗi index
        color_mask[mask == uid] = UCSD_PALETTE[int(uid)]
        
    return color_mask

def main():
    args = get_args()
    device = torch.device(args.device)

    print(f"Processing: {os.path.basename(args.image_path)}")

    # 1. Load Dữ liệu
    # Load Ảnh gốc
    img_pil = Image.open(args.image_path).convert("RGB")
    
    # Load Feature
    raw_feat_np = np.load(args.feature_path) # (C, H, W)
    C, H_feat, W_feat = raw_feat_np.shape
    
    # Resize ảnh gốc bằng kích thước feature để hiển thị cho khớp
    img_display = np.array(img_pil.resize((W_feat, H_feat)))

    # 2. Load Model & Chạy Rec(.)
    print("Running Rec(.)...")
    model = DVT(feature_dim=C, num_transformer_blocks=1)
    
    # Load weights (Xử lý state_dict)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()

    # Forward pass
    raw_tensor = torch.from_numpy(raw_feat_np).unsqueeze(0).float().to(device)
    with torch.no_grad():
        rec_tensor = model(raw_tensor)
    
    rec_feat_np = rec_tensor.squeeze(0).cpu().numpy() # (C, H, W)

    # 3. Tính PCA để Visualization
    print("Computing PCA...")
    pca_raw = compute_pca(raw_feat_np)
    pca_rec = compute_pca(rec_feat_np)

    # 4. Vẽ vời (Plotting)
    print("Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ảnh 1: Ảnh gốc
    axes[0].imshow(img_display)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Ảnh 2: Raw Feature (DINO)
    axes[1].imshow(pca_raw)
    axes[1].set_title("Raw Features (DINO)")
    axes[1].axis('off')

    # Ảnh 3: Rectified Feature (CoralSRT)
    axes[2].imshow(pca_rec)
    axes[2].set_title("Rectified Features (CoralSRT)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150)
    print(f"Saved visualization to: {args.save_path}")
    plt.show()

if __name__ == "__main__":
    main()
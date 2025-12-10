import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Create Target Features")
    parser.add_argument('--model_name', type=str, required=True,choices=["SAM","SAM_2","DINO_S_16","DINO_B_16","DINOv2_B_14","DINOv2_S_14","DINOv2_Reg","CoralSCOP"] ,help='Name of the model to use for feature extraction')
    parser.add_argument('--mode', type=str, choices=['average','median'],default='median', help='Method to compute target features')
    return parser.parse_args()

def process_image(args):
    feat_path, mask_path, save_path, mode = args
    feat_map = np.load(feat_path)
    C, H_feat, W_feat = feat_map.shape
    mask_img = Image.open(mask_path)
    mask_img = mask_img.resize((W_feat, H_feat), Image.NEAREST)
    mask_arr = np.array(mask_img)
    target_feat = feat_map.copy()
    flat_target = target_feat.reshape(C, -1)
    flat_mask = mask_arr.flatten()
    unique_ids = np.unique(mask_arr)
    for region_id in unique_ids:
        if region_id == 0:
            continue
        indices = np.where(flat_mask == region_id)[0]
        if len(indices) == 0:
            continue
        region_feats = flat_target[:, indices]
        if mode == 'average':
            region_stat = np.mean(region_feats, axis=1)
        else: 
            region_stat = np.median(region_feats, axis=1)
        for i in range(C):
            flat_target[i, indices] = region_stat[i]
    target_feat = flat_target.reshape(C, H_feat, W_feat)
    np.save(save_path, target_feat)

def main():
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
    elif model_name == "DINO_B_16":
        raw_feat_dir = "D:/Coral_SRT/data/raw_features/DINO_Base_16"
        if mode == 'average':
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINO_Base_16/mean"
        else:
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINO_Base_16/median"
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
    elif model_name == "DINOv2_Reg":
        raw_feat_dir = "D:/Coral_SRT/data/raw_features/DINOv2-Reg"
        if mode == 'average':
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINOv2-Reg/mean"
        else:
            target_feat_dir = "D:/Coral_SRT/data/target_features/DINOv2-Reg/median"
    elif model_name == "CoralSCOP":
        raw_feat_dir = "D:/Coral_SRT/data/raw_features/CoralSCOP"
        if mode == 'average':
            target_feat_dir = "D:/Coral_SRT/data/target_features/CoralSCOP/mean"
        else:
            target_feat_dir = "D:/Coral_SRT/data/target_features/CoralSCOP/median"
    feat_files = [f for f in os.listdir(raw_feat_dir) if f.endswith('.npy')]
    tasks = []
    for f_name in feat_files:
        base_name = os.path.splitext(f_name)[0]
        feat_path = os.path.join(raw_feat_dir, f_name)
        mask_path = os.path.join("D:/Coral_SRT/data/masks_png", base_name + '.png')
        save_path = os.path.join(target_feat_dir, f_name)
        if(os.path.exists(save_path)):
            continue
        tasks.append((feat_path, mask_path, save_path,mode))
    with Pool(processes=os.cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, tasks), total=len(tasks)))

if __name__ == "__main__":
    main()


import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from segment_anything import sam_model_registry
from sam2.build_sam import build_sam2
from vision_transformer import vit_huge, vit_large, vit_base, vit_small, vit_tiny, vit_base_r4, vit_small_patch14,ViTFeat

def get_args():
    parser = argparse.ArgumentParser(description="Extract features using Foudation Models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_file", type=str, required=True, help="File to save extracted features")
    parser.add_argument("--model_name", type=str, required=True,choices=["SAM","SAM_2","DINO_S_16","DINO_B_16","DINOv2_B_14","DINOv2_S_14","DINOv2_Reg","CoralSCOP"], help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    return parser.parse_args()

def load_sam_model(model_type, device):
    checkpoint_paths = "D:/Coral_SRT/checkpoints/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_paths)
    sam.to(device=device)
    return sam

def load_sam2_model(device):
    checkpoint = "D:/Coral_SRT/checkpoints/sam2_hiera_tiny.pt"
    config_path = "D:/Coral_SRT/.venv/Lib/site-packages/sam2/configs/sam2/sam2_hiera_t.yaml"
    sam2 = build_sam2(checkpoint=checkpoint, config_file=config_path, device=device)
    return sam2

def interpolate_pos_embed(state_dict, model):
    if 'pos_embed' not in state_dict:
        return state_dict
    if state_dict is None:
        return None
    pos_embed_checkpoint = state_dict['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_register_tokens = getattr(model, 'num_register_tokens', 0)
    num_extra_tokens_model = 1 + num_register_tokens
    grid_size_model = int(model.patch_embed.num_patches ** 0.5)
    n_ckpt = pos_embed_checkpoint.shape[1]
    grid_size_checkpoint = int((n_ckpt - num_extra_tokens_model) ** 0.5)
    num_extra_tokens_ckpt = num_extra_tokens_model
    
    if (grid_size_checkpoint ** 2 + num_extra_tokens_model) != n_ckpt:
        grid_size_checkpoint = int((n_ckpt - 1) ** 0.5)
        num_extra_tokens_ckpt = 1
    
    if grid_size_checkpoint == grid_size_model and num_extra_tokens_ckpt == num_extra_tokens_model:
        return state_dict
    
    extra_tokens_ckpt = pos_embed_checkpoint[:, :num_extra_tokens_ckpt, :]
    pos_tokens_ckpt = pos_embed_checkpoint[:, num_extra_tokens_ckpt:, :]
    if grid_size_checkpoint != grid_size_model:
        pos_tokens_ckpt = pos_tokens_ckpt.reshape(1, grid_size_checkpoint, grid_size_checkpoint, embedding_size).permute(0, 3, 1, 2)
        pos_tokens_ckpt = F.interpolate(
            pos_tokens_ckpt, 
            size=(grid_size_model, grid_size_model), 
            mode='bicubic', 
            align_corners=False
        )
        pos_tokens_ckpt = pos_tokens_ckpt.permute(0, 2, 3, 1).flatten(1, 2)
    
    if num_extra_tokens_ckpt != num_extra_tokens_model:
        new_extra_tokens = torch.zeros((1, num_extra_tokens_model, embedding_size), device=pos_embed_checkpoint.device, dtype=pos_embed_checkpoint.dtype)
        new_extra_tokens[:, 0:1, :] = extra_tokens_ckpt[:, 0:1, :]
        if num_extra_tokens_model > 1:
            new_extra_tokens[:, 1:, :] = extra_tokens_ckpt[:, 0:1, :].repeat(1, num_extra_tokens_model-1, 1)
        extra_tokens_ckpt = new_extra_tokens
    new_pos_embed = torch.cat((extra_tokens_ckpt, pos_tokens_ckpt), dim=1)
    state_dict['pos_embed'] = new_pos_embed
    return state_dict

def get_sam_feature_extractor(model, image, model_name, device):
    target_size = 1024
    w, h = image.size
    scale = target_size* 1.0 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    h_t, w_t = input_tensor.shape[2], input_tensor.shape[3]
    pad_h = target_size - h_t
    pad_w = target_size - w_t
    input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), value=0)
    features = None
    if model_name == "SAM":
        enc = model.image_encoder
        x = enc.patch_embed(input_tensor)
        if enc.pos_embed is not None:
            x = x + enc.pos_embed
        for blk in enc.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2)
        features = x
    elif model_name == "SAM_2":
        outputs = model.image_encoder(input_tensor)
        if isinstance(outputs, dict) and 'backbone_fpn' in outputs:
            feat_list = outputs['backbone_fpn']
            target_feat = None
            for feat in feat_list:
                if feat.shape[1] == 576:
                    target_feat = feat
                    break
            if target_feat is None:
                target_feat = feat_list[-1]
            features = target_feat
        else:
            features = outputs
    return features.cpu().numpy().squeeze(0)


def main():
    args = get_args()
    device = torch.device(args.device)

    is_sam = False
    model = None
    feat_extractor = None
    
    if args.model_name == "SAM":
        model = load_sam_model("vit_h", device)
        is_sam = True
    elif args.model_name == "SAM_2":
        model = load_sam2_model(device)
        is_sam = True
    elif args.model_name == "DINO_S_16":
        model = vit_small(patch_size=16, num_register_tokens=0)
        repo, ckpt = "facebookresearch/dino:main", "dino_vits16"
    elif args.model_name == "DINO_B_16":
        model = vit_base(patch_size=16, num_register_tokens=0)
        repo, ckpt = "facebookresearch/dino:main", "dino_vitb16"
    elif args.model_name == "DINOv2_B_14":
        model = vit_base(patch_size=14, num_register_tokens=0)
        repo, ckpt = "facebookresearch/dinov2", "dinov2_vitb14"
    elif args.model_name == "DINOv2_S_14":
        model = vit_small(patch_size=14, num_register_tokens=0)
        repo, ckpt = "facebookresearch/dinov2", "dinov2_vits14"
    elif args.model_name == "DINOv2_Reg":
        model = vit_base_r4(patch_size=14, num_register_tokens=4)
        repo, ckpt = "facebookresearch/dinov2", "dinov2_vitb14_reg"
    elif args.model_name == "CoralSCOP":
        return

    if not is_sam and 'repo' in locals():
        pretrained_model = torch.hub.load(repo, ckpt)
        state_dict = pretrained_model.state_dict() if hasattr(pretrained_model, 'state_dict') else pretrained_model
        state_dict = interpolate_pos_embed(state_dict, model)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        feat_extractor = ViTFeat(model, feat_dim=model.embed_dim, vit_feat="k", patch_size=model.patch_embed.patch_size)

    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    

    with torch.no_grad():
        for img_file in tqdm(image_files):
            base_name = os.path.basename(img_file).split('.')[0]
            save_path = os.path.join(args.output_file, base_name + '.npy')
            if os.path.exists(save_path):
                continue
        
            image_pil = Image.open(img_file).convert("RGB")
            
            if not is_sam:
                image_pil = image_pil.resize((518, 518), Image.BICUBIC)
        
            if is_sam:
                features = get_sam_feature_extractor(model, image_pil, args.model_name, device)
            else:
                img_tensor = transform_tensor(image_pil).unsqueeze(0).to(device)
                out = feat_extractor.model.get_intermediate_layers(img_tensor, n=1)[0]
                patch_size = model.patch_embed.patch_size
                H_feat = 518 // patch_size 
                W_feat = 518 // patch_size
                num_patches = H_feat * W_feat

                patch_tokens = out[:, -num_patches:, :] 
     
                B, N, Dim = patch_tokens.shape
                features = patch_tokens.transpose(1, 2).reshape(B, Dim, H_feat, W_feat)
                features = features.cpu().numpy().squeeze(0)

            print(f"File: {base_name} | Input: {image_pil.size} | Output: {features.shape}")
            np.save(save_path, features)

if __name__ == "__main__":
    main()

    

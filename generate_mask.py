import os
import json
import cv2
import torch
import numpy as np
from pycocotools import mask as mask_util
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

image = "D:/Coral_SRT/data/images"
exist_json = "D:/Coral_SRT/data/masks_json/existing"

checkpoint = "D:/Coral_SRT/checkpoints/sam2_hiera_tiny.pt"
config_path = "D:/Coral_SRT/.venv/Lib/site-packages/sam2/configs/sam2/sam2_hiera_t.yaml"
device = "cuda"

def imread_safe(path):
    try:
        stream = np.fromfile(path, np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("Error")
        return None

dtype = torch.float32

grid_size = 32
def create_grid_points(h,w,grid_size):
    x = np.linspace(0, w, grid_size)
    y = np.linspace(0, h, grid_size)
    xv, yv = np.meshgrid(x, y)
    points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    return points

sam2 = build_sam2(config_file=config_path, checkpoint=checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2)

def generate_masks(image, exist_json):
    image_files = [f for f in os.listdir(image)]
    with torch.inference_mode():
        for idx, image_file in enumerate(image_files):
            base_name = os.path.splitext(image_file)[0]
            json_name = base_name + ".json"
            path_exist = os.path.join(exist_json, json_name)
            if os.path.exists(path_exist):
                continue

            image_path = os.path.join(image, image_file)
            img = imread_safe(image_path)
            if img is None:
                print("Khong doc duoc")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            predictor.set_image(img_rgb)
            input_points = create_grid_points(height, width, grid_size)
            input_labels = np.ones(input_points.shape[0], dtype=int)
            masks_list = []
            batch_size = 32
            for i in range(0, len(input_points), batch_size):
                batch_points = input_points[i : i + batch_size]
                batch_labels = input_labels[i : i + batch_size]
                masks, scores, _ = predictor.predict(
                point_coords=batch_points,
                point_labels=batch_labels,
                multimask_output=False 
            )
                for m, s in zip(masks, scores):
                    if s > 0.4:
                        masks_list.append((m.squeeze(), s))
            
            print(f"-> Tìm thấy {len(masks_list)} masks ban đầu.")
            annotations=[]
            for i, (mask_raw, score) in enumerate(masks_list):
                mask_uint8 = mask_raw.astype(np.uint8)
                area = np.sum(mask_uint8)
                if area < 50: continue
                rle = mask_util.encode(np.asfortranarray(mask_uint8))
                rle['counts'] = rle['counts'].decode('utf-8')
                rows = np.any(mask_uint8, axis=1)
                cols = np.any(mask_uint8, axis=0)
                if not np.any(rows) or not np.any(cols): continue
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                anno={
                    'id': i,
                    'image_id': 0,
                    'area': float(area),
                    'segmentation': rle,
                    'bbox': bbox,
                }
                annotations.append(anno)
            output = {
                "image":{
                    "file_name": image_file,
                    "height": height,
                    "width": width,
                    "id": 0
                },
                "annotations": annotations}
            with open(path_exist, 'w') as f:
                json.dump(output, f)

if __name__ == "__main__":
    generate_masks(image, exist_json)

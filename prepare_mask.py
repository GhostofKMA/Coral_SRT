import glob
import os
import numpy as np
from PIL import Image
import json
import pycocotools.mask as mask
from multiprocessing import Pool
from tqdm import tqdm

input_image_dir = "D:/Coral_SRT/data/masks_json/existing"
output_mask_json_dir = "D:/Coral_SRT/data/masks_png"
num_processes = 4

def init_pool(num_processes, initializer=None, initargs=None):
    if initializer is None:
        return Pool(num_processes)
    elif initargs is None:
        return Pool(num_processes,initializer)
    else:
        if not isinstance(initargs, tuple):
            raise ValueError("initargs must be a tuple")
        return Pool(num_processes,initializer,initargs)
    
def process_image(data):
    json_path = data['json_path']
    save_path = data['mask_file']
    with open(json_path, 'r',encoding='utf-8') as f_labeled:
        a_labeled = json.load(f_labeled)

    img_info = a_labeled['image']
    width = img_info['width']
    height = img_info['height']
    annotations = a_labeled['annotations']

    min_area = 4096
    max_area = width * height*0.5
    output_annos=[]
    for item in annotations:
        area = item.get('area',0)
        if area > min_area and area < max_area:
            output_annos.append(item)

    output_annos.sort(key=lambda x: x['area'], reverse=True)
    item_mask = np.zeros((height, width), dtype=np.uint16)
    for idx, item in enumerate(output_annos):
        mask_arr = mask.decode(item['segmentation'])
        item_mask[mask_arr == 1] = idx + 1

    mask_img = Image.fromarray(item_mask.astype(np.uint8),"L")
    mask_img.save(save_path)


def track_parallel_progress(func,tasks,nproc=8):
    pool = Pool(nproc)
    results = []
    for r in tqdm(pool.imap_unordered(func, tasks), total=len(tasks)):
        results.append(r)

    pool.close()
    pool.join()
    return results

if __name__ == "__main__":
    json_files = glob.glob(os.path.join(input_image_dir, "*.json"))
    tasks = []
    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        mask_file = os.path.join(output_mask_json_dir, base_name + ".png")
        data = {
            'json_path': json_file,
            'mask_file': mask_file
        }
        tasks.append(data)

    track_parallel_progress(process_image, tasks, nproc=num_processes)
    
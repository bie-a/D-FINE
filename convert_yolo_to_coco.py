import os
import json
import cv2
from collections import defaultdict

# 类别映射（根据你的数据集实际类别调整）
CLASS_NAMES = [
    "c0 - Safe Driving", "c1 - Texting", "c2 - Talking on the phone",
    "c3 - Operating the Radio", "c4 - Drinking", "c5 - Reaching Behind",
    "c6 - Hair and Makeup", "c7 - Talking to Passenger"
]

def convert_yolo_to_coco(img_dir, label_dir, output_json):
    """
    YOLO 格式: 每行 <class_id> <x_center> <y_center> <width> <height> (归一化到 0~1)
    COCO 格式: 标准 COCO annotations JSON
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)]
    }
    
    ann_id = 0
    img_id = 0
    
    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 添加图片信息
        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })
        
        # 读取对应的 YOLO 标注
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    box_w = float(parts[3]) * w
                    box_h = float(parts[4]) * h
                    
                    # 转为 COCO 的 [x_min, y_min, width, height]
                    x_min = x_center - box_w / 2
                    y_min = y_center - box_h / 2
                    
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [round(x_min, 2), round(y_min, 2), 
                                 round(box_w, 2), round(box_h, 2)],
                        "area": round(box_w * box_h, 2),
                        "iscrowd": 0
                    })
                    ann_id += 1
        
        img_id += 1
    
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"Saved {output_json}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

if __name__ == "__main__":
    # 根据你的实际路径修改
    convert_yolo_to_coco(
        img_dir="data/train/images",
        label_dir="data/train/labels",
        output_json="dataset/annotations/instances_train.json"
    )
    convert_yolo_to_coco(
        img_dir="data/valid/images",
        label_dir="data/valid/labels",
        output_json="dataset/annotations/instances_val.json"
    )
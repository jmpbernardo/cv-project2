import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch
from ultralytics import YOLO
from transformers import pipeline
from torchvision.datasets import VOCDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
warnings.filterwarnings('ignore')

def prepare_targets(subset, targets_list):
    targets_list=[]
    for img, target in subset:
        objs = target['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
        boxes = []
        labels = []
        for obj in objs:
            b = obj['bndbox']
            boxes.append([int(b['xmin']), int(b['ymin']), int(b['xmax']), int(b['ymax'])])
            labels.append(VOC_CLASS_TO_ID[obj['name']])
        targets_list.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        })
    return targets_list

def convert_yolo_predictions(yolo_results, coco_classes, voc_map):
    yolo_boxes = torch.tensor(yolo_results.boxes.xyxy, dtype=torch.float32)
    yolo_scores = torch.tensor(yolo_results.boxes.conf, dtype=torch.float32)

    # Convert YOLO class indices → names → VOC ids
    labels_list = []
    for label_idx in yolo_results.boxes.cls:
        name = coco_classes[label_idx.item()]
        label_id = voc_map.get(name, 0)
        labels_list.append(label_id)

    yolo_labels = torch.tensor(labels_list, dtype=torch.int64)

    return {
        "boxes": yolo_boxes,
        "scores": yolo_scores,
        "labels": yolo_labels,
    }

def convert_detr_predictions(detr_results, voc_map, confidence=0.1):
    # Filter by confidence
    detr_results = [r for r in detr_results if r["score"] >= confidence]

    if len(detr_results) == 0:
        return None

    boxes_list = []
    scores_list = []
    labels_list = []

    for r in detr_results:
        box = r["box"]

        # Convert to xyxy
        if isinstance(box, dict):
            boxes_list.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
        else:
            x, y, w, h = box
            boxes_list.append([x, y, x + w, y + h])

        scores_list.append(r["score"])
        label_id = voc_map.get(r["label"], 0)
        labels_list.append(label_id)

    return {
        "boxes": torch.tensor(boxes_list, dtype=torch.float32),
        "scores": torch.tensor(scores_list, dtype=torch.float32),
        "labels": torch.tensor(labels_list, dtype=torch.int64),
    }

def evaluate_sample(img, target, yolo_model, detr_pipeline, coco_classes, voc_map,
                    metric_yolo, metric_detr, confidence):
    img_np = np.array(img)

    # --- YOLO ---
    yolo_raw = yolo_model(img_np, conf=confidence, verbose=False)[0]
    yolo_preds = convert_yolo_predictions(yolo_raw, coco_classes, voc_map)
    metric_yolo.update([yolo_preds], [target])

    # --- DETR ---
    detr_raw = detr_pipeline(img)
    detr_preds = convert_detr_predictions(detr_raw, voc_map, confidence=confidence)

    if detr_preds is not None:
        metric_detr.update([detr_preds], [target])


def evaluate_models(subset, targets_list, voc_map, confidence):
    # Load models
    yolo_model = YOLO("yolo11s.pt")
    detr_pipeline = pipeline("object-detection", model="facebook/detr-resnet-50")

    COCO_CLASSES = yolo_model.names

    results = {
        'yolo': {},
        'detr': {}
    }

    for conf in confidence:
        metric_yolo = MeanAveragePrecision()
        metric_detr = MeanAveragePrecision()

        # Loop through samples
        for sample, target in zip(subset, targets_list):
            img = sample[0]
            evaluate_sample(img, target, yolo_model, detr_pipeline,
                            COCO_CLASSES, voc_map, metric_yolo, metric_detr, conf)

        results['yolo'][conf] = metric_yolo.compute()
        results['detr'][conf] = metric_detr.compute()

        # print(f"  YOLO mAP: {results['yolo'][conf]['map']:.4f}")
        # print(f"  DETR mAP: {results['detr'][conf]['map']:.4f}")

    return results

if __name__ == "__main__":
    dataset = VOCDetection('data/', year="2007", image_set="val", download=False)
    subset = [dataset[i] for i in range(1000)]

    # Mapping VOC classes to IDs
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    VOC_CLASS_TO_ID = {cls_name: idx + 1 for idx, cls_name in enumerate(VOC_CLASSES)} #TODO: é suposto ser array a começar em 1

    targets_list = prepare_targets(subset, VOC_CLASS_TO_ID)

    confidence = [0.001, 0.05, 0.1, 0.3, 0.5]
    results = evaluate_models(subset, targets_list, VOC_CLASS_TO_ID, confidence)
    for conf in confidence:
        print(f"\nConfidence Threshold: {conf}")
        print(f"  YOLO - mAP: {results['yolo'][conf]['map']:.4f}, mAP@50: {results['yolo'][conf]['map_50']:.4f}")
        print(f"  DETR - mAP: {results['detr'][conf]['map']:.4f}, mAP@50: {results['detr'][conf]['map_50']:.4f}")
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
from transformers import DetrForObjectDetection, DetrImageProcessor

warnings.filterwarnings('ignore')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_targets(subset):
    targets_list = []
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


def convert_detr_predictions(detr_results, coco_classes, voc_map):
    labels_list = []
    for l in detr_results["labels"]:
        name = coco_classes[l.item()]
        label_id = voc_map.get(name, 0)
        labels_list.append(label_id)

    return {
        "boxes": detr_results['boxes'],
        "scores": detr_results['scores'],
        "labels": torch.tensor(labels_list, dtype=torch.int64)
    }


def convert_rtdetr_predictions(rtdetr_results, coco_classes, voc_map):
    labels_list = []
    for l in rtdetr_results["labels"]:
        name = coco_classes[l.item()]
        label_id = voc_map.get(name, 0)
        labels_list.append(label_id)

    return {
        "boxes": rtdetr_results['boxes'],
        "scores": rtdetr_results['scores'],
        "labels": torch.tensor(labels_list, dtype=torch.int64)
    }


def evaluate_yolo(yolo_model, subset, targets_list, voc_map, conf):
    metric_yolo = MeanAveragePrecision()

    for sample, target in zip(subset, targets_list):
        img = sample[0]
        img_np = np.array(img)
        yolo_raw = yolo_model(img_np, conf=conf, verbose=False)[0]
        yolo_preds = convert_yolo_predictions(yolo_raw, yolo_model.names, voc_map)
        metric_yolo.update([yolo_preds], [target])

    return metric_yolo.compute()


def evaluate_detr(detr_processor, detr_model, subset, targets_list, voc_map, conf):
    metric_detr = MeanAveragePrecision()

    for sample, target in zip(subset, targets_list):
        img = sample[0]
        img_np = np.array(img)

        inputs = detr_processor(images=img_np, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = detr_model(**inputs)

        height, width = img_np.shape[:2]

        detr_raw = detr_processor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=conf)[0]

        detr_preds = convert_detr_predictions(detr_raw, detr_model.config.id2label, voc_map)

        metric_detr.update([detr_preds], [target])

    return metric_detr.compute()


def evaluate_rtdetr(rtdetr_image_processor, rtdetr_model, subset, targets_list, voc_map, conf):
    metric_rtdetr = MeanAveragePrecision()

    for sample, target in zip(subset, targets_list):
        img = sample[0]
        img_np = np.array(img)

        inputs = rtdetr_image_processor(img_np, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = rtdetr_model(**inputs)

        height, width = img_np.shape[:2]
        rtdetr_raw = rtdetr_image_processor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=conf
        )[0]
        rtdetr_preds = convert_rtdetr_predictions(rtdetr_raw,
                                                  rtdetr_model.config.id2label, voc_map)
        metric_rtdetr.update([rtdetr_preds], [target])

    return metric_rtdetr.compute()


def evaluate_models(subset, targets_list, voc_map, confidence):
    results = {
        'yolo': {},
        'detr': {},
        'rt-detr': {}
    }
    yolo_model = YOLO("yolo11s.pt").to(device)

    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

    rtdetr_image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
    rtdetr_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd").to(device)

    for conf in confidence:
        results['yolo'][conf] = evaluate_yolo(yolo_model, subset, targets_list, voc_map, conf)
        results['detr'][conf] = evaluate_detr(detr_processor, detr_model, subset, targets_list, voc_map, conf)
        results['rt-detr'][conf] = evaluate_rtdetr(rtdetr_image_processor, rtdetr_model,
                                                   subset, targets_list, voc_map, conf)

    return results


def print_metrics(results, confidence):
    for conf in confidence:
        print(f"Confidence Threshold: {conf} \n")

        for model_name in ['yolo', 'detr', 'rt-detr']:
            result = results[model_name][conf]

            print(f"\n{model_name.upper()} Metrics:")
            print(f"  mAP :  {result['map']:.4f}")
            print(f"  mAP@50:    {result['map_50']:.4f}")
            print(f"  mAP@75 :    {result['map_75']:.4f}")
            print(f"  mAP (small objects):  {result['map_small']:.4f}")
            print(f"  mAP (medium objects): {result['map_medium']:.4f}")
            print(f"  mAP (large objects):  {result['map_large']:.4f}")
            print(f"  mAR@1:                {result['mar_1']:.4f}")
            print(f"  mAR@10:               {result['mar_10']:.4f}")
            print(f"  mAR@100:              {result['mar_100']:.4f}")
            print(f"  mAR (small objects):  {result['mar_small']:.4f}")
            print(f"  mAR (medium objects): {result['mar_medium']:.4f}")
            print(f"  mAR (large objects):  {result['mar_large']:.4f}")


if __name__ == "__main__":

    set_seed(42)
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Fetching the device that will be used throughout this notebook
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device", device)

    dataset = VOCDetection('data/', year="2007", image_set="val", download=False)
    subset = [dataset[i] for i in range(1000)]  # first 1000 samples

    # Mapping VOC classes to IDs
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    VOC_CLASS_TO_ID = {cls_name: idx + 1 for idx, cls_name in
                       enumerate(VOC_CLASSES)}

    targets_list = prepare_targets(subset)

    confidence = [0.001, 0.05, 0.1, 0.3, 0.5]
    results = evaluate_models(subset, targets_list, VOC_CLASS_TO_ID, confidence)

    print_metrics(results, confidence)

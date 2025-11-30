

import torch
from torchvision.datasets import CocoDetection
import numpy as np


class TACODETRDetectionDataset(torch.utils.data.Dataset):

    # We're using the COCO dataset class from torchvision to load the TACO dataset
    # Each time an image is requested (__getitem__), we apply the transformations and 
    # invoke the processor to prepare the image and annotations

    def __init__(self, img_folder, ann_file, processor, transform=None):
        self.coco_dataset = CocoDetection(img_folder, ann_file)
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.coco_dataset)

    # PyTorch will invoke this method to get a sample by index
    def __getitem__(self, idx):
        image, annotations = self.coco_dataset[idx]

        boxes = [ann["bbox"] for ann in annotations]
        category_ids = [ann["category_id"] for ann in annotations]
        
        # Convert image to RGB numpy array
        image = np.array(image.convert("RGB"))

        
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category=category_ids)
            image = transformed["image"]
            new_boxes = transformed["bboxes"]
            new_ids = transformed["category"]

            # Reconstruct annotations for the Processor
            # The processor expects the same structure (List[Dict]) but with updated coordinates
            updated_annotations = []
            for i, box in enumerate(new_boxes):
                # Overwriting bbox and category_id with the transformed versions
                new_ann = annotations[i].copy()
                new_ann["bbox"] = box
                new_ann["category_id"] = new_ids[i]
                updated_annotations.append(new_ann)
            
            annotations = updated_annotations

        # Wrap annotations in the format expected by the processor
        target = {
            "image_id": idx,
            "annotations": annotations
        }

        # Processor handles resizing, normalization, target conversion
        result = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        # Image processor expands batch dimension, lets squeeze it
        result = {k: v[0] for k, v in result.items()}

        return result



def taco_detr_collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data
import os
import json
import re
import numpy as np

class DataFormatter():
    def __init__(self, data):
        self.data = data
        self.objects_category_map = self.category_mapping()
        for k, v in self.objects_category_map.items():
            print(f"Objects Category Mapping: {k} -> {v}")

    def category_mapping(self):
        sorted_objects = sorted(self.data.all_objects_collection, key=lambda x: x.collection_name.lower())
        objs_map = {f.collection_name: idx + 1 for idx, f in enumerate(sorted_objects)}      
        return objs_map
    
    def clip_bounding_box(self, bbox):
        x, y, a, b = bbox
        x = max(0, min(self.data.resx, x))
        a = max(0, min(self.data.resx, a))
        y = max(0, min(self.data.resy, y))
        b = max(0, min(self.data.resy, b))      
        return (x, y, a, b)

    def format_bounding_box_to_YOLO(self, bbox):
        bbox = self.clip_bounding_box(bbox)
        x, y, a, b = bbox
        xcenter = ((x + a)/2)/self.data.resx
        ycenter = ((y+b)/2)/self.data.resy
        width = (a-x)/self.data.resx
        height = (b-y)/self.data.resy
        return xcenter, ycenter, width, height

    def format_bounding_box_to_COCO(self, bbox):
        bbox = self.clip_bounding_box(bbox)
        x, y, a, b = bbox
        width = a-x
        height = b-y
        return x, y, width, height
    
    def clip_keypoints(self, keypoint):
        x, y = keypoint
        x = max(0, min(self.data.resx, x))
        y = max(0, min(self.data.resy, y))
        return (x, y)
    
    def format_keypoints_to_COCO(self, keypoints):
        keypoints_coco = []
        for k in keypoints:
            x, y = self.clip_keypoints((k["x"], k["y"]))
            v = 2
            if k["occluded"]:
                v = 1
            if not k["inFrame"]:
                v = 0
            keypoints_coco.extend([x,y,v])
        return keypoints_coco

    def format_keypoints_to_YOLO(self, keypoints):
        keypoints_yolo = []
        for k in keypoints:
            x, y = self.clip_keypoints((k["x"], k["y"]))
            x = x/self.data.resx
            y = y/self.data.resy
            v = 2
            if k["occluded"]:
                v = 1
            if not k["inFrame"]:
                v = 0
            keypoints_yolo.append((x,y,v))
        return keypoints_yolo
    
    def get_name_from_value(self, mapping, value):
        return next((k for k, v in mapping.items() if v == value), None)

    
    def get_bbox_area(self, bbox):
        x, y, w, h = self.format_bounding_box_to_COCO(bbox)
        return w * h
    

    def export_data_COCO(self, file, saveAfterIterations):
        coco_data = None
        superCategory = "InventoryItems"
        running_annotation_id = 0
        if os.path.exists(file):
            with open(file, "r") as f:
                coco_data = json.load(f)
            print(f"Json file opened: {file}")
            try:
                lastid = coco_data["annotations"][-1]["id"]
                running_annotation_id = lastid + 1
            except Exception as e:
                print("Could not extract last annotation id", e)

        else:
            coco_data = {
                "info":
                {
                    "description": "Inventory tiny dataset",
                    "url": "unspecified",
                    "version": "1.0",
                    "year": 2025,
                    "contributor": "Sabal Dahal",
                    "date_created": "2025/08/07"
                },
                "licenses": 
                {
                    "id": 1,
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "name": "CC BY 4.0"
                },
                "categories": 
                [
                    {
                        "id": 0,
                        "name": superCategory,
                        "supercategory": "none"
                    },
                    *(
                        {
                            "id": self.objects_category_map[obj_collection.collection_name],
                            "name": obj_collection.collection_name,
                            "supercategory": superCategory
                        } for obj_collection in self.data.all_objects_collection
                    )
                ],
                "images": [],
                "annotations": []
            }
        
        totalSaved = 0
        while True:
            data = yield
            if isinstance(data, bool) and data:
                with open(file, "w") as f:
                    json.dump(coco_data, f, indent=4)
                print(f"[FINAL SAVE] All annotations saved to {file}")
                return
            
            image_index, bbox = data

            
            coco_data["images"].append(
                {
                    "id": image_index,
                    "file_name": f"{image_index:06d}.png",
                    "width": self.data.resx,
                    "height": self.data.resy
                }
            )
            coco_data["annotations"].extend(
                [
                    {
                        "id": running_annotation_id,
                        "image_id": image_index,
                        "category_id": self.objects_category_map[name],
                        "bbox": list(self.format_bounding_box_to_COCO(values)),
                        "area": self.get_bbox_area(values),
                        "segmentation": [],
                        "iscrowd": 0
                    }
                    for i, (name, values) in enumerate(bbox.items())
                ]
            )
            running_annotation_id += 1
            totalSaved += 1
            if totalSaved >= saveAfterIterations:
                with open(file, "w") as f:
                    json.dump(coco_data, f, indent=4)
                print(f"[AUTO SAVE] saved file {file}")
                totalSaved = 0
  
    def export_data_YOLO(self, label_dir, image_index, bbox, keypoints):
        xcenter, ycenter, width, height = self.format_bounding_box_to_YOLO(bbox)
        keypoints_yolo = self.format_keypoints_to_YOLO(keypoints)
        yolo_line = f"0 {xcenter} {ycenter} {width} {height}"
        for k in keypoints_yolo:
            x, y, v = k
            yolo_line = yolo_line + f" {x} {y} {v}"        
        label_path = os.path.join(label_dir, f"{image_index:06d}.txt")
        with open(label_path, "w") as f:
            f.write(yolo_line + "\n")

    def export_transformation_matrix(self, dir, image_index, matrix):
        file = os.path.join(dir, f"{image_index:06d}.txt")
        np.savetxt(file, matrix, fmt="%.7f")

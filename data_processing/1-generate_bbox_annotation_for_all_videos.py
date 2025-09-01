from ultralytics import YOLO
import os
import json
import numpy as np
from collections import OrderedDict


model = YOLO('your yolo path')  # 预训练的 YOLOv8n 模型

file_names = os.listdir("your cholect50 annotation file name")
file_dir = "your cholect50 annotation file dir"

for file in file_names:
    file_path = os.path.join(file_dir,file)
    with open(file_path) as f:
        dataset = json.load(f)
    video = dataset["video"]
    for k,v in dataset["annotations"].items():
        #transform your video files into frames
        img_file = "path to video/CholecT50/videos/VID{:02}/{:06}.png".format(int(video),int(k))
        results = model.predict(img_file,conf=0.5,iou=0.5)
        for result in results:
            boxes = result.boxes.xywhn.cpu().numpy().tolist()
            labels = result.boxes.cls.cpu().numpy()
            annotated_list = []
            for index,box in enumerate(boxes):
                label = int(labels[index])
                for list_index,i in enumerate(v):
                    if i[1] == label and list_index not in annotated_list:
                        i[3:7] = box
                        annotated_list.append(list_index)
                        break
                    else:
                        continue
    annotation = dataset["annotations"]
    sorted_keys = sorted(annotation.keys(), key=int)
    sorted_data = OrderedDict()
    for key in sorted_keys:
        sorted_data[key] = annotation[key]
    dataset["annotations"] = sorted_data
    with open(file_path,"w") as f:
        f.write(json.dumps(dataset))
   

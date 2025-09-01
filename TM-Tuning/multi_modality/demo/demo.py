import numpy as np
import os
import io
import cv2

import torch

from config import (Config,
                    eval_dict_leaf)

from utils import (retrieve_text,
                  _frame_from_video,
                  setup_internvideo2)


video = cv2.VideoCapture('example video path')
frames = [x for x in _frame_from_video(video)]

text_candidates = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
                   "The endoscopic video depicts a laparoscopic cholecystectomy procedure. During the \"preparation\" phase, a grasper surgical instrument is actively retracting the gallbladder. The grasper's position varies across the frames, being located around the center, moving closer to the top-right, and making dynamic adjustments throughout the section.",
                   "The video depicts a laparoscopic cholecystectomy surgery during the calot-triangle-dissection phase. Throughout the video, a grasper is frequently seen retracting the gallbladder, while a hook engages in dissection of the gallbladder. The grasper is consistently positioned near the lower-center of the field, and the hook's location is not within the visible field. The surgical instruments are actively engaged in these tasks, and the procedures smoothly progress without any idle time noted for the instruments.",
                   "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
                   "The endoscopic video features a laparoscopic cholecystectomy surgery in its preparation phase. Throughout the footage, a grasper instrument is consistently seen retracting the gallbladder. Initially, the grasper is located near the lower-right area of the screen, later moving towards the center. In subsequent frames, the exact location of the grasper in the frame becomes unclear.",
                   "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
                   "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
                   "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
                   "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]


config = Config.from_file('internvideo2_stage2_config.py')
config = eval_dict_leaf(config)

model_pth = 'trained model path'
config['pretrained_path'] = model_pth
config["model"]["vision_encoder"]["pretrained"] = model_pth

config.model.text_encoder["config"] = "./configs/config_bert_large.json"

config.model.text_encoder["pretrained"] = "bert-large-uncased"

intern_model, tokenizer = setup_internvideo2(config)

texts, probs = retrieve_text(frames, text_candidates, model=intern_model, topk=5, config=config)

for t, p in zip(texts, probs):
    print(f'text: {t} ~ prob: {p:.4f}')
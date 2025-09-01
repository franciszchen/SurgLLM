"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import math
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import CaptionDataset
import numpy as np

VIDEO_TEMPLATE = [
"What is this video?",
"Tell me about this video."
"A short caption for the video:",
"A short description of the video:",
"A video that shows",
"Describe the video briefly.",
"Write a description for the video.",
"Provide a description of what is presented in the video.",
"Briefly describe the content of the video.",
"Can you briefly explain what you see in the video?",
"Could you use a few words to describe what you perceive in the video?",
"Please provide a short description of the video.",
"Using language, provide a short account of the video.",
"Use a few words to illustrate what is happening in the video.",
"Summarize the video content briefly:",
"Give a brief overview of the video:",
"What does the video depict?",
"What is shown in the video?",
"Explain the video content in short:",
"Can you summarize what happens in the video?",
"What is portrayed in the video?",
"Briefly describe what the video is about:",
"What can be seen in the video?",
"Describe briefly what is happening in the video:",
"Provide a quick description of the video:",
"What does the video illustrate?",
"Give a short summary of the video:",
"What occurs in the video?",
"What is the video showing?",
"What is the main content of the video?",
"How would you describe the video content?",
"Can you give a brief account of the video?",
"What is the video about?",
"Describe the events in the video in brief:",
"What is depicted in the video?",
"What happens in the video?",
"What is the focus of the video?",
"What is the main theme of the video?",
"What scenes are presented in the video?",
"What is the primary subject of the video?",
"Briefly explain the video content:",
"What are the key moments in the video?",
"What is the storyline of the video?",
"Provide a brief summary of the video:",
"What is the essence of the video?",
"What actions are shown in the video?",
"What visual content does the video contain?",
"What is the main idea of the video?",
"What is the core message of the video?",
"What is captured in the video?",
"What visuals are presented in the video?",
"What is the video illustrating?",
"Can you describe the video scene?",
"What is the overall content of the video?",
"What highlights are shown in the video?",
"What details are in the video?",
"What narrative is the video presenting?",
"What is the video focusing on?",
"What picture does the video paint?",
"What is the subject matter of the video?",
"What is the key content of the video?",
"What is the video demonstrating?",
"What is the video explaining?",
"What is the plot of the video?",
"What story is told in the video?",
"What is the theme of the video?",
"What information does the video convey?",
"What is the gist of the video?",
"What does the video document?",
"What is shown visually in the video?",
"What is the video presentation about?",
"What is the video narrative?",
"What is the video sequence about?",
"What is the video content?",
"What is the video footage showing?",
"What events are in the video?",
"What is the video scenario?",
"What scenes are depicted in the video?",
"What is the video recording about?",
"What is the video clip about?",
"What visuals does the video depict?",
"What is the video segment about?",
"What is the overview of the video?",
"What does the video feature?",
"What is the video exhibition?",
"What is the video describing?",
"What is highlighted in the video?",
"What is the video material?",
"What is the brief of the video?",
"What is the visual story in the video?",
"What is the video exploration?",
"What moments are in the video?",
"What is the video presentation?",
"What is the video showcase?",
"What is the video coverage?",
"What is the video reporting?",
"What is depicted visually in the video?",
"What is the video illustration?",
"What is the video portrayal?",
"What is the video evidence?",
"What is the video recording?",
"What is the video documentation?",
"What is the video depiction?",
"What is the video exhibit?",
"What is the video demonstration?",
"What is the video showcase about?",
"What is the video narrative about?",
"What is the video commentary?",
"What is the visual content of the video?",
"What is the video imagery?",
"What does the video illustrate visually?",
"What is the focus of the video content?",
"How would you summarize the video?",
"What is the video's main point?",
"Outline the contents of the video briefly.",
"What does the video show?",
"Can you describe the scenes in the video?",
"What key information does the video provide?",
"Give a concise description of the video.",
"What are the main events in the video?",
"What story does the video tell?",
"Can you outline what the video is about?",
"Provide a quick summary of what the video covers.",
"Explain the primary focus of the video.",
"What do you observe in the video?",
"Summarize the key points of the video in a few sentences.",
"What are the highlights of the video?",
"Describe what you see in the video in a few words.",
"What is the purpose of the video?",
"Can you detail the contents of the video?",
"What does the video highlight?",
"Provide an overview of the video content.",
"What themes are explored in the video?",
"Can you give a snapshot of the video?",
"What are the visual elements shown in the video?",
"Describe the narrative of the video.",
"What are the essential parts of the video?",
"How would you explain the video to someone who hasn't seen it?",
"What is the video's storyline?",
"Can you provide a brief explanation of the video's content?",
"What critical scenes are depicted in the video?",
"Explain what is captured in the video.",
"Describe the video in a nutshell.",
"What are the core messages of the video?",
"Can you describe the setting of the video?",
"What action takes place in the video?",
"How does the video begin and end?",
"What is the climax of the video?",
"Describe the main characters in the video.",
"What is the conflict or challenge shown in the video?",
"What resolution is presented in the video?",
"How does the video contribute to the understanding of its topic?",
"What is the emotional impact of the video?",
"What visual style is used in the video?",
"What audio elements accompany the video?",
"Can you describe any symbolic elements in the video?",
"What questions does the video raise?",
"How does the video relate to its intended audience?",
"What is the pace of the video?",
"Describe any notable cinematography in the video.",
"What editing techniques are evident in the video?",
"What genres does the video encompass?",
"What is unique about the video?",
"How do the visuals support the video's message?",
"What feedback or reactions does the video aim to provoke?",
"Describe the opening scene of the video.",
"What are the technical aspects of the video?",
"How is dialogue used in the video?",
"What are the underlying themes in the video?",
"Describe any metaphors found in the video.",
"What type of language is used in the video?",
"How is the video structured?",
"What are the video's strengths and weaknesses?",
"What lessons can be learned from the video?",
"What is the historical context of the video?",
"Describe any controversies associated with the video.",
"What is the professional quality of the video?",
"How current is the information in the video?",
"What are the educational aspects of the video?",
"How does the video compare to similar works?",
"What are the ethical considerations in the video?",
"Describe the production values of the video.",
"What cultural aspects are explored in the video?",
"How does the video address its themes?",
"What are the persuasive elements of the video?",
"Describe any innovative aspects of the video.",
"What is the target demographic of the video?",
"How does the video fit into its genre?",
"What are the most impactful moments in the video?",
"Explain any controversial points in the video.",
"Describe the climax of the video in detail.",
"What are the implications of the video's message?",
"How does the video contribute to its field?",
"What future discussions could the video inspire?",
"Describe the video in terms of its cultural significance.",
"What are the artistic elements of the video?",
"How does the video use color and lighting?",
"What are the soundtracks used in the video?",
"Describe the most surprising moment in the video.",
"What predictions can you make about the video's impact?",
"How does the video connect with its viewers?",
"What are the follow-up questions after watching the video?"
]


class CholecVideoCaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        try:
            #Can do video load and tranform here, save [B,T,C,W,H]
            video = video_path
        except:
            print(f"Could not load {video_path}")
            return None
        if video==None:
            return None
        
        caption = self.text_processor(ann["caption"])
        query = np.random.choice(VIDEO_TEMPLATE)


        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": query,
            "text_output":caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class VideoCaptionEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # videos set. do not repeat videos in inference
        ## todo: make it deduplicated because creating annotation file makes 
        seen = set()
        self.annotation = [x for x in self.annotation if x["video"] not in seen and not seen.add(x["image_id"])]
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        try:
            video = self.vis_processor(video_path)
        except:
            print(f"Could not load {video_path}")
            return None

        return {
            "video": video,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }


# class VideoCaptionInstructDataset(VideoCaptionDataset):
#     def __getitem__(self, index):
#         data = super().__getitem__(index)
#         if data != None:
#             data['text_output'] = data["text_input"]
#             data['text_input'] = self.text_processor("")
#         return data



class ClipCaptionDataset(BaseDataset):
    """
    Handles video datasets where subclip of full video needs to be loaded. 
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video_path"]
        video_path = os.path.join(self.vis_root, vname)
        try:
            video = self.vis_processor(video_path, start_sec=math.floor(ann['ts'][0]), end_sec=math.ceil(ann['ts'][1]))
        except:
            return None


        caption = ann["caption"] if 'caption' in ann else ann["query"]

        image_id = ann['youtube_id'] if 'youtube_id' in ann else ann["video_id"] if "video_id" in ann else vname

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": self.text_processor(caption),
            "image_id": image_id,
            "instance_id": ann['instance_id'],
        }

class ClipCaptionInstructDataset(ClipCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data

class ClipCaptionEvalDataset(ClipCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data


class WebVideoCaptionDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    def _get_video(self, index):
        """
        If video does not exist, loop to the next one.
        """
        max_retries = 3
        for _ in range(max_retries):
            ann = self.annotation[index]
            video_path = os.path.join(self.vis_root, f"{ann['videoid']}.mp4")
            try:
                video = self.vis_processor(video_path)
                return video, video_path, ann
            except:
                index = (index + 1) % len(self.annotation)  # Safely loop back to start of annotations
        return None

    def __getitem__(self, index):
        video, video_path, ann = self._get_video(index)
        caption = self.text_processor(ann["name"])

        # "image_id" is kept for compatibility with the COCO evaluation format
        return {
            "video": video,
            "text_input": caption,
            "image_id": ann["videoid"],
            "instance_id": ann["instance_id"],
        }

class WebVideoCaptionInstructDataset(WebVideoCaptionDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = data["text_input"]
            data['text_input'] = self.text_processor("")
        return data

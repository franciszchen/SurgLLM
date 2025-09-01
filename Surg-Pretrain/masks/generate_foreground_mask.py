import json
import numpy as np
import os
import math
from decord import VideoReader, cpu


def check_overlap(patch_left, patch_top, patch_right, patch_bottom, box_left, box_top, box_right, box_bottom):
    """检查patch是否与bounding box重叠"""
    # 如果有任何不重叠的情况，返回0
    if (patch_right <= box_left or patch_left >= box_right or
        patch_bottom <= box_top or patch_top >= box_bottom):
        return 0
    return 1

def generate_patch_list(bbox, grid_size=16):
    """
    生成与bounding box重叠的patch列表。
    
    :param bbox: tuple, 表示bounding box的 (x_min, y_min, x_max, y_max)
    :param grid_size: 图片分割的行和列数量，默认为16
    :return: list, 长度为256，其中重叠的patch为1，不重叠的为0
    """
    x_min, y_min, x_max, y_max = bbox
    patch_list = []
    patch_width = 1.0 / grid_size
    patch_height = 1.0 / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算当前patch的边界坐标
            patch_left = j * patch_width
            patch_top = i * patch_height
            patch_right = patch_left + patch_width
            patch_bottom = patch_top + patch_height
            
            # 检查当前patch是否与bounding box重叠
            overlap = check_overlap(patch_left, patch_top, patch_right, patch_bottom, x_min, y_min, x_max, y_max)
            patch_list.append(overlap)
    
    return patch_list


def convert_bbox_format(bbox):
    """
    将边界框从 [top_left_x, top_left_y, box_width, box_height] 格式
    转换为 [top_left_x, top_left_y, bottom_right_x, bottom_right_y] 格式。

    参数:
    bbox : list
        包含 [top_left_x, top_left_y, box_width, box_height] 的列表。

    返回:
    list
        包含 [top_left_x, top_left_y, bottom_right_x, bottom_right_y] 的列表。
    """
    top_left_x, top_left_y, box_width, box_height = bbox
    bottom_right_x = top_left_x + box_width
    bottom_right_y = top_left_y + box_height
    
    # 确保坐标值在0到1之间
    bottom_right_x = min(max(bottom_right_x, 0), 1)
    bottom_right_y = min(max(bottom_right_y, 0), 1)
    
    return [top_left_x, top_left_y, bottom_right_x, bottom_right_y]



if __name__ == "__main__":

    json_dir = "cholect50-challenge-val/labels"
    json_files = os.listdir(json_dir)
    video_dir = "videos"
    save_path = "mask.json"
    all_file_dict = {}
    for file in json_files:
        all_frames_dict = {}
        file_path = os.path.join(json_dir,file)
        video_path = video_dir+"/"+file.split(".")[0]+".mp4"
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        total_frames = len(vr)

        with open(file_path,"r") as f:
            dataset = json.load(f)
        
        key_frames = dataset["annotations"]
        first_idx = [i for i in key_frames.keys()][0]

        nearest_frame = first_idx
        print("detect first index : ",first_idx," for video: ",file)
        for i in range(1,total_frames+1):
            if str(i) in key_frames:
                nearest_frame = str(i)
            all_frames_dict[str(i)] = [x[3:7] for x in key_frames[nearest_frame]]
        
        file_name = file.split(".")[0]

        for i,v in all_frames_dict.items():
            total_mask = np.zeros(256)
            for bbox in v:
                xyxy_bbox = convert_bbox_format(bbox)
                mask = np.array(generate_patch_list(xyxy_bbox))
                total_mask = np.logical_or(total_mask,mask).astype(int).tolist()
            all_frames_dict[i] = total_mask

        all_file_dict[file_name] = all_frames_dict

    with open(save_path,"w") as f:
        f.write(json.dumps(all_file_dict,indent=2))

    


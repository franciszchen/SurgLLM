import argparse
import json
import os

import openai
import time

        
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

if __name__ == '__main__':
    output_path = "annotations_from_step_1"
    video_dir = "video/CholecT50/videos/"
    json_dir = "output_file_name"
    
    json_files = os.listdir(json_dir)
    file_list = []
    red_flag = False
    for json_file in json_files:
        print(json_file)
        with open(os.path.join(json_dir,json_file),"r") as f:
            data = json.load(f)
        vid = json_file.split(".")[0]
        name_map = data["categories"]
        phase_map = name_map["phase"]
        triplet_map = name_map["triplet"]

        
        for k,v in data["annotations"].items():
            item_dict = {}
            frame_id = str(k)
            frame_path = os.path.join(video_dir,vid,frame_id)
            item_dict["file_path"] = frame_path
            info_list = []
            for i in v:
                if i[3]>=0:
                    info_dict = {} 
                    if i[0]<0:
                        info_dict["triplet"] = "Null"
                    else:
                        info_dict["triplet"] = triplet_map[str(i[0])]
                    info_dict["instrument_location"] = convert_bbox_format(i[3:7])
                    item_dict["phase"] = phase_map[str(i[14])].replace("-"," ")
                    info_list.append(info_dict)
                else:
                    info_dict = {} 
                    if i[0]<0:
                        info_dict["triplet"] = "Null"
                    else:
                        info_dict["triplet"] = triplet_map[str(i[0])]
                    info_dict["instrument_location"] = i[3:7]
                    item_dict["phase"] = phase_map[str(i[14])].replace("-"," ")
                    info_list.append(info_dict)

            item_dict["annotation"] = info_list
            if not red_flag:
                file_list.append(item_dict)
            red_flag = False
    with open(output_path,"w") as f:
        f.write(json.dumps(file_list,indent=2))


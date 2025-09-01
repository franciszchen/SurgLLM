import argparse
import json
import os

import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5
import os
from openai import OpenAI
from tqdm import tqdm
openai_api_key = "EMPTY"
openai_api_base = ""
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)



def get_result(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=1.25
    )
    return response.choices[0].message.content



def sliding_window_list(index_list, window_size, stride):
    return [index_list[i:i+window_size] for i in range(0, len(index_list) - window_size + 1, stride)]

def is_sorted_ascending(lst):
    for i in range(len(lst) - 1):
        if int(lst[i]) > int(lst[i + 1]):
            return False
    return True

    
if __name__ == '__main__':
    output_path = "GPT_CAP_SHORT.json"
    caption_list = []
    with open("annotation_locations.json","r") as f:
        annotation_locations = json.load(f)
    with open("annotation_phase.json","r") as f:
        annotation_phase = json.load(f)
    with open("annotation_triplet.json","r") as f:
        annotation_triplet = json.load(f)


    prompt = """
    ## Character
    You are a surgical AI visual assistant, and you are seeing an endoscopic video from a laparoscopic cholecystectomy surgery. 
    ## Scene relation
    You are seeing a triplet list, denoting the object relations in the consecutive endoscopy frames, This triplet list contains several dictionaries, each dictionary illustrates a triplet (surgical instrument, verb, target), null means idle as follows:\n
    ```json
    {triplet}
    ```

    ## Location
    You are seeing a location list in the consecutive frames that contains instrument localization bounding box scaled from 0 to 1(top left x, top left y, bottom right x, bottom right y) of multiple frames within the current second, inside each frame it contains several dictionaries, and each dictionary illustrates its location, as follows:\n
    ```json
    {locations}
    ```

    ## Surgical phase
    You are seeing the current surgical phase in the consecutive frames as follows:\n"
    ```json
    {phase}
    ```

    ## Task
    Based on these facts, Your task is to generate a brief description about the video using the previous given information within 100 words.


    ## Constraints
    -   It should be a description of one video not frames.
    -   Do not mention any specific number of location, a rough position like "center", "top right" is enough.
    -	Do not make up any thing without solid evidence in the given information dictionaries.
    -	Importantly, you do not need to give any reasoning process, just give a straightforward description.
    -   Do not mention specific time in the answer.
    """
    records = []
    previous_data_path = ""
    if os.path.exists(previous_data_path):
        with open(previous_data_path, "r") as f:
            previous_data = json.load(f)
        for record in previous_data:
            unique_id = record["folder"]+str(record["seg_sec"])
            records.append(unique_id)

    data_length = len(annotation_phase)
    index_list = [i for i in range(len(annotation_phase))]
    window_list = sliding_window_list(index_list,4,16)
    
    for i in tqdm(window_list): 
        video_cap = {}
        window_cap_list = []
        window_phase_list = []
        window_location_list = []
        window_triplet_list = []
        video_cap["folder"] = annotation_phase[i[0]]["file_path"].split("/")[-2]
        seg_sec = []
        for index in i:
            cap_dict = {}
            cap_dict["current_phase"] = annotation_phase[index]["phase"]
            cur_sec = int(annotation_phase[index]["file_path"].split("/")[-1])
            seg_sec.append(cur_sec)
            cap_dict["current_second"] = cur_sec-min(seg_sec)+1
            cap_dict["phase"] = annotation_phase[index]["phase"]
            cap_dict["locations"] = annotation_locations[index]["location"]
            cap_dict["triplet"] = annotation_triplet[index]["triplet"]
            window_phase_list.append(cap_dict["phase"])
            window_location_list.append(cap_dict["locations"])
            window_triplet_list.append(cap_dict["triplet"])
            unique_id = video_cap["folder"]+str(seg_sec)
            if unique_id in records:
                print(unique_id," is already in the json! ")
                continue
        if is_sorted_ascending(seg_sec):
            final_prompt = prompt.format(triplet=window_triplet_list,locations= window_location_list,phase=window_phase_list)
            answer = get_result(final_prompt)
            video_cap["caption"] = answer
            video_cap["seg_sec"] = seg_sec
            video_cap["tag"] = "cap_short"
           
        else:
            print("not from same folder")
            continue
        
        caption_list.append(video_cap)


        with open(output_path, "w") as f:
             f.write(json.dumps(caption_list,indent=2))
        
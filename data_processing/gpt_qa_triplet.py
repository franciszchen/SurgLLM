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
    # 遍历列表中的每个元素，除了最后一个
    for i in range(len(lst) - 1):
        # 如果当前元素大于下一个元素，则不是升序
        if int(lst[i]) > int(lst[i + 1]):
            return False
    return True

    
if __name__ == '__main__':
    output_path = "GPT_QA_TRIPLET.json"
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
    Based on these facts, Your task is to generate several questions related to the triplet.

    ### Few Examples
    -	"Question: What is the grasper doing during the video? Answer: The grasper at first is retracting the liver, then it is retracting the gallbladder",
    -	"Question: What is the hook doing? Answer: The hook is dissecting the gallbladder."
   -	"Question: During the video, what are the instruments and what are they doing? Answer: The grasper is grasping the gallbladder and the clipper is clipping cystic duct ."
   -	"Question: Is the hook dissecting the gallbladder? Answer: Yes ."
   -	"Question: Is the grasper grasping the gallbladder? Answer: No, it is retracting the gallbladder ."



    ## Constraints
    -	You can ask questions with diversity.
    -   Do not mention any specific number of location, a rough position like "center", "top right" is enough.
    -	Remember, all the questions can be clearly answered based on the information in the given lists. 
    -	Do not make up any questions and answers without solid evidence in the given information dictionaries.
    -	Importantly, you do not need to give any reasoning process, just give a straightforward answer.
    -   Do not mention specific time in the answer.
    """

    
    data_length = len(annotation_phase)
    index_list = [i for i in range(len(annotation_phase))]
    window_list = sliding_window_list(index_list,16,96)
    
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
        if is_sorted_ascending(seg_sec):
            final_prompt = prompt.format(triplet=window_triplet_list,locations= window_location_list,phase=window_phase_list)
            #print(final_prompt)
            answer = get_result(final_prompt)
            #print(video_cap["folder"]," answer: ",answer)
            video_cap["caption"] = answer
            video_cap["seg_sec"] = seg_sec
            video_cap["tag"] = "qa_triplet"
           
        else:
            print("not from same folder")
            continue
        
        caption_list.append(video_cap)


        with open(output_path, "w") as f:
             f.write(json.dumps(caption_list,indent=2))
        
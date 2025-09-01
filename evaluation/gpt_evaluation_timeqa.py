from openai import OpenAI
import os
import json
import os
import numpy as np
import base64
from tqdm import tqdm
import time
openai_api_key = "EMPTY"
openai_api_base = ""

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id



prompt_head = """You are now an automated grading system tasked with calcu answers based on provided ground truth answers(gt_answer) and predicted answers(pred_answer). 
                Here are the gt answer and predicted answer."""
prompt_end ="""Ensure accuracy in your assessments, and do not offer any explanations for why each answer is correct or incorrect. 
            The answer might related to a correct time, deviations of 1-5 seconds are allowed.
            The answer might have similar meanings, no need to be the same, if they have a similar meaning, it is also correct.
            You only need to give 1 for correct answers and 0 for wrong answers, no any other word is needed in your answer.
            """



def get_result(prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0
    )
    return response.choices[0].message.content


    
if __name__ == '__main__':
    dir_ = "evaluation/videollm"
    path =  "videollm/eval_timeqa.json"
    output_path =os.path.join(dir_,path)
    caption_list = []
    count = 0
    with open("data path","r") as f:
        data = json.load(f)

    
    for i in tqdm(data[:345]):
        temp_dict = {}
        final_prompt = prompt_head+" ground truth answer: "+str(i["gt_answer"])+" predicted answer: "+str(i["pred_answer"])+prompt_end
        score = get_result(final_prompt)
        # temp_dict["id"] = i["id"]
        temp_dict["score"] = score


        caption_list.append(temp_dict)
        count+=1

        with open(output_path, "w") as f:
             f.write(json.dumps(caption_list,indent=2))

        
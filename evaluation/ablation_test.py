from tqdm import tqdm
import argparse
import os
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import glob
import time
from openai import OpenAI

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

# OpenAI设置
openai_api_key = "your api key"
openai_api_base = "your api link"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# GPT评分提示词
prompt_head = """You are now an automated grading system tasked with evaluating answers based on provided ground truth answers(gt_answer) and predicted answers(pred_answer). 
                Here are the gt answer and predicted answer."""
prompt_end = """Ensure accuracy in your assessments, and do not offer any explanations for why each answer is correct or incorrect. 
            The answer might related to a correct time, deviations of 1-5 seconds are allowed.
            The answer might have similar meanings, no need to be the same, if they have a similar meaning, it is also correct.
            You only need to give 1 for correct answers and 0 for wrong answers, no any other word is needed in your answer.
            """


class Args:
    """模拟argparse返回的对象"""
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.options = []


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls


def get_dataset_type(folder_name):
    """根据文件夹名称获取对应的数据集类型"""
    dataset_keywords = ['location', 'movement', 'phase', 'relation', 'triplet', 'timeqa']
    
    for keyword in dataset_keywords:
        if keyword in folder_name.lower():
            return keyword
    
    return None


def get_test_data(test_dataset, spliter,dataset_type):
    """根据数据集类型返回对应的测试数据"""
    if dataset_type == 'location':
        return test_dataset[:spliter]
    elif dataset_type == 'movement':
        return test_dataset[spliter:spliter*2]
    elif dataset_type == 'phase':
        return test_dataset[spliter*2:spliter*3]
    elif dataset_type == 'relation':
        return test_dataset[spliter*3:spliter*4]
    elif dataset_type == 'triplet':
        return test_dataset[spliter*4:]
    elif dataset_type == 'timeqa':
        # timeqa使用全部数据
        return test_dataset
    else:
        return None


def load_test_dataset(dataset_type):
    """根据数据集类型加载对应的测试数据"""
    if dataset_type == 'timeqa':
        test_file = "timeqa_test.json"
    else:
        test_file = "test.json"
    
    with open(test_file, "r") as f:
        test_dataset = json.load(f)
    
    return test_dataset


def get_gpt_result(prompt):
    """进行评分"""
    try:
        response = client.chat.completions.create(
            model="your model path",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT评分出错: {str(e)}")
        return None


def evaluate_with_gpt(data, output_path):
    """对预测结果进行评分"""
    print(f"\n开始进行评分...")
    caption_list = []
    count = 0
    
    for i in tqdm(data, desc="评分进度"):
        temp_dict = i.copy()  # 保留原始数据
        final_prompt = prompt_head + " ground truth answer: " + str(i["gt_answer"]) + " predicted answer: " + str(i["pred_answer"]) + prompt_end
        
        # 获取GPT评分，如果失败则重试
        max_retries = 3
        for retry in range(max_retries):
            score = get_gpt_result(final_prompt)
            if score is not None:
                break
            if retry < max_retries - 1:
                print(f"重试第 {retry + 1} 次...")
                time.sleep(2)  # 等待2秒后重试
        
        temp_dict["score"] = score if score is not None else "error"
        caption_list.append(temp_dict)
        count += 1
        
        # 定期保存结果，避免意外中断导致数据丢失
        if count % 10 == 0:
            with open(output_path, "w") as f:
                f.write(json.dumps(caption_list, indent=2))
    
    # 最终保存
    with open(output_path, "w") as f:
        f.write(json.dumps(caption_list, indent=2))
    
    # 计算准确率
    correct = sum(1 for item in caption_list if item.get("score") == "1")
    total = len(caption_list)
    accuracy = correct / total if total > 0 else 0
    
    print(f"评分完成！正确数: {correct}, 总数: {total}, 准确率: {accuracy:.2%}")
    return accuracy


def process_ablation_model(ablation_folder_path, ablation_folder_name,spliter):
    """处理单个消融模型"""
    
    # 提取frame数量（如2f, 4f等）
    frame_num = ablation_folder_name.split('_')[0]  # 获取2f, 4f等
    
    # 构建cfg路径
    cfg_path = f"checkpoint/finetune_lora_qa_{frame_num}.yaml"
    
    # 创建模拟的args对象
    args = Args(cfg_path)
    
    # 创建配置
    cfg = Config(args)
    
    # 设置任务和模型
    task = tasks.setup_task(cfg)
    print(f"\n处理消融模型: {ablation_folder_name}")
    print(f"使用配置文件: {cfg_path}")
    print("初始化模型...")
    model = task.build_model(cfg)
    
    # 构建full parameter model路径
    full_para_model_path = ""
    print(f"加载完整参数模型: {full_para_model_path}")
    model.load_from_pretrained(full_para_model_path)
    
    # 查找checkpoint_4.pth文件
    checkpoint_path = None
    for root, dirs, files in os.walk(ablation_folder_path):
        if 'checkpoint_4.pth' in files:
            checkpoint_path = os.path.join(root, 'checkpoint_4.pth')
            break
    
    if checkpoint_path is None:
        print(f"警告: 在 {ablation_folder_path} 中未找到 checkpoint_4.pth")
        return
    
    print(f"加载LoRA模型: {checkpoint_path}")
    model.load_from_pretrained(checkpoint_path)
    
    # 获取对应的数据集类型
    dataset_type = get_dataset_type(ablation_folder_name)
    if dataset_type is None:
        print(f"警告: 无法识别 {ablation_folder_name} 对应的数据集类型")
        return
    
    # 加载对应的测试数据集
    test_dataset = load_test_dataset(dataset_type)
    print(f"加载测试数据集: {'timeqa_test.json' if dataset_type == 'timeqa' else 'test.json'}")
    
    # 获取对应的测试数据
    test_data = get_test_data(test_dataset, spliter, dataset_type)
    if test_data is None:
        print(f"警告: 无法获取 {dataset_type} 对应的测试数据")
        return
    
    print(f"测试数据集类型: {dataset_type}, 数据量: {len(test_data)}")
    
    # 进行测试
    answer_list = []
    model.eval()
    
    for i in tqdm(test_data, desc=f"测试 {ablation_folder_name}"):
        answer_dict = {}
        answer_dict["question"] = i["question"]
        answer_dict["video"] = i["video"]
        answer_dict["id"] = i["image_id"]
        answer_dict["gt_answer"] = i["answer"]
        
        input_sample = {"video": [i["video"]], "prompt": i["question"]}
        pred_answer = model.generate(input_sample)[0]
        answer_dict["pred_answer"] = pred_answer
        
        answer_list.append(answer_dict)
    
    # 保存原始结果
    output_filename = f"{ablation_folder_name}.json"
    with open(output_filename, "w") as f:
        f.write(json.dumps(answer_list, indent=2))
    
    print(f"结果已保存到: {output_filename}")
    
    # 使用GPT进行评分
    gpt_output_filename = f"{ablation_folder_name}_gpt.json"
    accuracy = evaluate_with_gpt(answer_list, gpt_output_filename)
    
    # 清理GPU内存
    del model
    torch.cuda.empty_cache()
    
    return ablation_folder_name, accuracy


def main():
    job_id = now()
    
    # 初始化分布式模式
    # 创建一个默认的args对象用于初始化
    default_args = Args("")
    default_cfg = Config(default_args)
    spliter = 101
    init_distributed_mode(default_cfg.run_cfg)
    setup_seeds(default_cfg)
    setup_logger()
    
    # 定义ablation文件夹路径
    ablation_base_path = ""
    
    # 获取所有消融模型文件夹
    ablation_folders = [d for d in os.listdir(ablation_base_path) 
                       if os.path.isdir(os.path.join(ablation_base_path, d))]
    
    print(f"找到 {len(ablation_folders)} 个消融模型文件夹")
    print(f"消融模型列表: {ablation_folders}")
    
    # 存储所有模型的准确率
    all_results = {}
    
    # 遍历每个消融模型
    for ablation_folder_name in ablation_folders:
        ablation_folder_path = os.path.join(ablation_base_path, ablation_folder_name)
        
        try:
            model_name, accuracy = process_ablation_model(ablation_folder_path, ablation_folder_name,spliter)
            all_results[model_name] = accuracy
        except Exception as e:
            print(f"处理 {ablation_folder_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印所有模型的准确率汇总
    print("\n========== 所有消融模型测试完成！==========")
    print("准确率汇总：")
    for model_name, accuracy in all_results.items():
        print(f"{model_name}: {accuracy:.2%}")
    
    # 保存汇总结果
    summary_file = "ablation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n准确率汇总已保存到: {summary_file}")


if __name__ == "__main__":
    main()
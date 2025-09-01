import cv2
import os
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import pandas as pd
import io
from collections import Counter
import re
import numpy as np
from PIL import Image

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()
    return frame_count

def check_videos(directory,frames):
    # check all videos that less than certain frames
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 支持的视频格式
            full_path = os.path.join(directory, filename)
            frame_count = count_frames(full_path)
            if frame_count < frames:
                print(f"视频 {filename} 的帧数少于16帧（实际帧数：{frame_count}帧）。")
            else:
                pass


def generate_word_cloud(json_path):
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # 读取JSON数据
    with open(json_path,"r") as f:
        json_data = json.load(f)

    # 提取所有caption文本
    captions_text = " ".join([item["caption"] for item in json_data])

    # 自定义停用词
    custom_stopwords = set(stopwords.words('english'))
    custom_stopwords.update(['the', 'and', 'of', 'to', 'is', 'in', 'for', 'a', 'this', 'are', 'as', 'by', 'on', 'with', 'that'])

    # 分词并计算词频
    words = re.findall(r'\b\w+\b', captions_text.lower())
    filtered_words = [word for word in words if word not in custom_stopwords and len(word) > 2]
    word_counts = Counter(filtered_words)

    # 打印最常见的词汇
    print("最常见的医学术语：")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")

    highlighted_word = "video"  
    highlighted_count = 50000      

    # 创建一个新的词频字典
    custom_word_counts = word_counts.copy()
    custom_word_counts[highlighted_word] = highlighted_count

    # 创建词云
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        stopwords=custom_stopwords,
        max_words=100,
        colormap='viridis',
        collocations=False,
        contour_width=3,
        contour_color='steelblue'
    ).generate_from_frequencies(custom_word_counts)

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    # 保存图片
    plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')

    # 显示词云
    plt.show()


def check_video_frame(video_path):
# 加载视频文件
    video_path = video_path
    cap = cv2.VideoCapture(video_path)


    # 使用CAP_PROP_FRAME_COUNT来获取视频的总帧数
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total number of frames in the video: {frames}')

    # 释放视频文件
    cap.release()


def create_video_from_images(start,end,video_folder, output_video_path, fps=1):
    # 获取图片文件列表并排序
    images = []
    for i in range(start,end+1):
        images.append("/ssd/lxj/CholecT50/videos/VID{:02}/{:06}.png".format(video_folder,i))
    # 读取第一张图片以获取视频的尺寸 
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
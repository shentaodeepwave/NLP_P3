#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
from collections import Counter
import matplotlib.pyplot as plt  # 新增导入

def load_data(filename):
    """加载数据"""
    words = []
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            doublet = line.strip().split("\t")
            if len(doublet) < 2:
                continue
            words.append(doublet[0])
            labels.append(doublet[1])
    return words, labels

def count_common_names(filename, top_n=100):
    """统计最常见的人名"""
    words, labels = load_data(filename)
    name_counter = Counter()

    for word, label in zip(words, labels):
        if label == "PERSON":
            name_counter[word] += 1
    print(f"统计到 {len(name_counter)} 个名字")
    
    # 获取最常见的前 top_n 个人名
    most_common_names = name_counter.most_common(top_n)
    
    # 计算前 top_n 名字的总出现次数
    top_n_count = sum(count for _, count in most_common_names)
    # 计算所有人名的总出现次数
    total_count = sum(name_counter.values())
    
    # 计算比例
    percentage = (top_n_count / total_count) * 100
    print(f"前 {top_n} 个名字的出现次数占总人名数的比例为: {percentage:.2f}%")
    
    return name_counter

def save_common_names_to_json(common_names, output_file):
    """保存常见名字到 JSON 文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dict(common_names), f, ensure_ascii=False, indent=4)

def plot_name_distribution(name_counter, interval=100):
    """绘制人名分布折线图（纵坐标为累计出现比例）"""
    sorted_names = name_counter.most_common()
    counts = [count for _, count in sorted_names]
    
    # 计算总出现次数
    total_count = sum(counts)
    
    # 每 interval 个名字统计一次累计出现比例
    cumulative_counts = [sum(counts[:i]) for i in range(interval, len(counts) + 1, interval)]
    cumulative_ratios = [count / total_count * 100 for count in cumulative_counts]
    x_labels = list(range(interval, len(counts) + 1, interval))
    
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, cumulative_ratios, marker='o', linestyle='-', color='b')
    plt.title("cumulative proportion of occurrences of people's names")
    plt.xlabel("number of names of people")
    plt.ylabel("Cumulative percentage of occurrences (%)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    train_data_path = "./data/train"
    output_file = "./common_names.json"
    top_n = 1000

    name_counter = count_common_names(train_data_path, top_n)
    save_common_names_to_json(name_counter.most_common(top_n), output_file)
    print(f"最常见的 {top_n} 个名字已保存到 {output_file}")
    
    # 绘制折线图
    plot_name_distribution(name_counter, interval=100)
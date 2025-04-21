#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json
from collections import Counter

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
    
    return most_common_names

def save_common_names_to_json(common_names, output_file):
    """保存常见名字到 JSON 文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dict(common_names), f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    train_data_path = "./data/train"
    output_file = "./common_names.json"
    top_n = 1000

    common_names = count_common_names(train_data_path, top_n)
    save_common_names_to_json(common_names, output_file)
    print(f"最常见的 {top_n} 个名字已保存到 {output_file}")
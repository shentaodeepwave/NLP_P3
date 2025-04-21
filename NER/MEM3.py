#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from nltk.classify.maxent import MaxentClassifier
from sklearn.metrics import (accuracy_score, fbeta_score, precision_score, recall_score)
import os
import nltk
import pickle
from nltk import pos_tag
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.tokenize import sent_tokenize  # 添加分句所需的模块
from concurrent.futures import ThreadPoolExecutor

nltk.download('punkt')

# 下载 NLTK 所需的资源
nltk.download('averaged_perceptron_tagger')

import json

class MEMM():
    def __init__(self):
        self.train_path = "./data/train"
        self.dev_path = "./data/dev"
        self.beta = 0
        self.max_iter = 0
        self.classifier = None

    def features(self, words, previous_label, position, tagged_words=None):
        """提取特征"""
        features = {}
        current_word = words[position]
        
        # 确保 current_word 是字符串
        if not isinstance(current_word, str):
            current_word = str(current_word)
        
        features['has_(%s)' % current_word] = 1
        features['prev_label'] = previous_label
        if current_word[0].isupper():
            features['case=Title'] = 1
        if current_word.isupper():
            features['case=ALLCAP'] = 1
        if "'" in current_word:
            features['format=Apostrophe'] = 1
        n = 1
        if position > 0:
            features[f'prev_{n}_word={words[position - 1]}'] = 1
        if position < len(words) - 1:
            features[f'next_{n}_word={words[position + 1]}'] = 1
        # 使用缓存的词性标注结果
        if tagged_words:
            features[f'pos={tagged_words[position][1]}'] = 1
        features[f'length={len(current_word)}'] = 1
        if any(char.isdigit() for char in current_word):
            features['has_digit=1'] = 1
        if any(not char.isalnum() for char in current_word):
            features['has_special_char=1'] = 1 
        return features
    
    def load_data(self, filename):
        """加载数据并按句号分句"""
        sentences = []
        sentence_labels = []
        words = []
        labels = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                doublet = line.strip().split("\t")
                if len(doublet) < 2:
                    continue
                word, label = doublet
                words.append(word)
                labels.append(label)
                if word == "." or word == "?" or word == "!":
                    # 遇到句号、问号或感叹号，保存当前句子并清空
                    sentences.append(words)
                    sentence_labels.append(labels)
                    words = []
                    labels = []
            # 如果最后还有未保存的句子
            if words:
                sentences.append(words)
                sentence_labels.append(labels)
        return sentences, sentence_labels


    def extract_features_for_sentence(self, words, labels):
        """为单个句子提取特征"""
        previous_labels = ["O"] + labels
        tagged_words = pos_tag(words)  # 对句子进行词性标注
        features = [
            self.features(words, previous_labels[i], i, tagged_words=tagged_words)
            for i in range(len(words))
        ]
        return features

    def train(self):
        """训练分类器（多线程版）"""
        print('Training classifier...')
        sentences, sentence_labels = self.load_data(self.train_path)
        train_samples = []

        # 使用多线程提取特征
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.extract_features_for_sentence, sentences, sentence_labels))

        # 将特征和标签组合成训练样本
        for features, labels in zip(results, sentence_labels):
            train_samples.extend([(f, l) for (f, l) in zip(features, labels)])

        classifier = MaxentClassifier.train(train_samples, max_iter=self.max_iter)
        self.classifier = classifier

    def test(self):
        """测试分类器（多线程版）"""
        print('Testing classifier...')
        sentences, sentence_labels = self.load_data(self.dev_path)
        all_results = []
        all_labels = []

        # 使用多线程提取特征
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.extract_features_for_sentence, sentences, sentence_labels))

        # 分类并收集结果
        for features, labels in zip(results, sentence_labels):
            results = [self.classifier.classify(n) for n in features]
            all_results.extend(results)
            all_labels.extend(labels)

        # 计算评估指标
        f_score = fbeta_score(all_labels, all_results, average='macro', beta=self.beta)
        precision = precision_score(all_labels, all_results, average='macro')
        recall = recall_score(all_labels, all_results, average='macro')
        accuracy = accuracy_score(all_labels, all_results)
        print("%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n" % (
            "f_score=", f_score, "accuracy=", accuracy, "recall=", recall, "precision=", precision))
        return True
    def dump_model(self):
        """保存模型"""
        with open('./model.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self):
        """加载模型"""
        with open('./model.pkl', 'rb') as f:
            self.classifier = pickle.load(f)



    def predict_sentence(self, text):
        """对输入的文本进行命名实体识别，支持多句子"""
        sentences = sent_tokenize(text)  # 分句
        all_named_entities = []

        print("Step-by-step prediction process:")
        for sentence in sentences:
            words = word_tokenize(sentence)  # 使用 NLTK 的 word_tokenize 进行分词
            previous_labels = ["O"]
            tagged_words = pos_tag(words)  # 对句子进行词性标注

            features = []
            results = []
            named_entities = []

            for i in range(len(words)):
                # 提取特征
                feature = self.features(words, previous_labels[i], i, tagged_words=tagged_words)
                features.append(feature)

                # 分类
                result = self.classifier.classify(feature)
                results.append(result)
                previous_labels.append(result)

                # 如果是命名实体，添加到结果中
                is_person = result == "PERSON"
                named_entities.append((words[i], is_person))

            # 打印当前句子的人名
            print(f"Sentence: {sentence}")
            print("Named Entities:", [word for word, is_person in named_entities if is_person])

            # 保存当前句子的命名实体结果
            all_named_entities.append(named_entities)

        return all_named_entities
    
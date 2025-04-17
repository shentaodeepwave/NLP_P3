#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from nltk.classify.maxent import MaxentClassifier
from sklearn.metrics import (accuracy_score, fbeta_score, precision_score, recall_score)
import os
import nltk
import pickle
from nltk import pos_tag
from tqdm import tqdm

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
        with open('./common_names.json', 'r', encoding='utf-8') as f:
            self.common_names = set(json.load(f).keys())
    def features(self, words, previous_label, position, tagged_words=None):
        """提取特征"""
        features = {}
        current_word = words[position]
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
        if current_word in self.common_names:
            features['is_common_name=1'] = 1
        features[f'length={len(current_word)}'] = 1
        if any(char.isdigit() for char in current_word):
            features['has_digit=1'] = 1
        if any(not char.isalnum() for char in current_word):
            features['has_special_char=1'] = 1
        return features

    def load_data(self, filename):
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

    def train(self):
        """训练分类器"""
        print('Training classifier...')
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        tagged_words = pos_tag(words)  

        features = [
            self.features(words, previous_labels[i], i, tagged_words=tagged_words)
            for i in tqdm(range(len(words)), desc="Extracting Features")
        ]

        train_samples = [(f, l) for (f, l) in zip(features, labels)]
        classifier = MaxentClassifier.train(train_samples, max_iter=self.max_iter)
        self.classifier = classifier

    def test(self):
        """测试分类器"""
        print('Testing classifier...')
        words, labels = self.load_data(self.dev_path)
        previous_labels = ["O"] + labels
        tagged_words = pos_tag(words)  # 对整个句子进行词性标注

        features = [
            self.features(words, previous_labels[i], i, tagged_words=tagged_words)
            for i in range(len(words))
        ]
        results = [self.classifier.classify(n) for n in features]

        # 计算评估指标
        f_score = fbeta_score(labels, results, average='macro', beta=self.beta)
        precision = precision_score(labels, results, average='macro')
        recall = recall_score(labels, results, average='macro')
        accuracy = accuracy_score(labels, results)
        print("%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n" % (
            "f_score=", f_score, "accuracy=", accuracy, "recall=", recall, "precision=", precision))
        return True

    def show_samples(self, bound):
        """显示样本预测结果"""
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        tagged_words = pos_tag(words)  # 对整个句子进行词性标注

        features = [
            self.features(words, previous_labels[i], i, tagged_words=tagged_words)
            for i in range(len(words))
        ]
        (m, n) = bound
        pdists = self.classifier.prob_classify_many(features[m:n])
        print('  Words          P(PERSON)  P(O)\n' + '-' * 40)
        for (word, label, pdist) in list(zip(words, labels, pdists))[m:n]:
            if label == 'PERSON':
                fmt = '  %-15s *%6.4f   %6.4f'
            else:
                fmt = '  %-15s  %6.4f  *%6.4f'
            print(fmt % (word, pdist.prob('PERSON'), pdist.prob('O')))

    def dump_model(self):
        """保存模型"""
        with open('./model.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self):
        """加载模型"""
        with open('./model.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
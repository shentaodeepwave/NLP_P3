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
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import sent_tokenize  # 添加分句所需的模块
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
        with open('./common_names.json', 'r', encoding='utf-8') as f:
            self.common_names = set(json.load(f).keys())

    def features(self, words, previous_label, position, tagged_words=None):
        """改进的特征提取"""
        features = {}
        current_word = words[position]

        # 确保 current_word 是字符串
        if not isinstance(current_word, str):
            current_word = str(current_word)

        # 当前单词的基本特征
        features['word.isupper'] = current_word.isupper()  # 是否全大写
        features['word.istitle'] = current_word.istitle()  # 是否首字母大写
        features['word.isalpha'] = current_word.isalpha()   # 是否是字母
        features['word.numeric'] = current_word.isnumeric()  # 是否是数字
        features['word.length'] = len(current_word)  # 单词长度
        features['word.has_digit'] = any(char.isdigit() for char in current_word)  # 是否包含数字
        features['word.has_special_char'] = any(not char.isalnum() for char in current_word)  # 是否包含标点符号
        features['word.has_Apostrophe'] = True if "'" in current_word else False  # 是否包含撇号
        features['word.suffix'] = current_word[-3:] if len(current_word) > 2 else current_word  # 后缀
        features['word.prefix'] = current_word[:3] if len(current_word) > 2 else current_word  # 前缀

        # 是否是常见名字
        features['is_common_name'] = current_word in self.common_names

        # 上下文单词特征
        if position > 0:
            prev_word = words[position - 1]
            features['prev_word.istitle'] = prev_word.istitle()
            features['prev_word.isupper'] = prev_word.isupper()
            features['prev_word.isalpha'] = prev_word.isalpha()
        else:
            features['BOS'] = True  # 句子开头

        if position < len(words) - 1:
            next_word = words[position + 1]
            features['next_word.istitle'] = next_word.istitle()
            features['next_word.isupper'] = next_word.isupper()
            features['next_word.isalpha'] = next_word.isalpha()
            features['next_word.is_end_punctuation'] = next_word in {'.', '?', '!'}  # 检测后一个单词是否是结束句子的标点符号
        else:
            features['EOS'] = True  # 句子结尾

        # 词性特征
        if tagged_words:
            features['pos'] = tagged_words[position][1]  # 当前单词的词性
            if position > 0:
                features['prev_pos'] = tagged_words[position - 1][1]  # 前一个单词的词性
            if position < len(tagged_words) - 1:
                features['next_pos'] = tagged_words[position + 1][1]  # 后一个单词的词性

        # 前一个标签
        features['prev_label'] = previous_label

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
        return features, labels

    def train(self):
        """训练分类器，使用多线程提取特征"""
        print('Training classifier...')
        sentences, sentence_labels = self.load_data(self.train_path)
        train_samples = []

        # 使用多线程提取特征
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.extract_features_for_sentence, sentences, sentence_labels))

        # 整理训练样本
        for features, labels in results:
            train_samples.extend([(f, l) for (f, l) in zip(features, labels)])

        classifier = MaxentClassifier.train(train_samples, max_iter=self.max_iter)
        self.classifier = classifier

    def test(self):
        """测试分类器，使用多线程提取特征"""
        print('Testing classifier...')
        sentences, sentence_labels = self.load_data(self.dev_path)
        all_results = []
        all_labels = []

        # 使用多线程提取特征
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.extract_features_for_sentence, sentences, sentence_labels))

        # 分类并收集结果
        for features, labels in results:
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

    def show_samples(self, bound):
        """显示所有样本的预测结果"""
        sentences, sentence_labels = self.load_data(self.dev_path)
        all_words = []
        all_labels = []
        all_features = []

        # 遍历所有句子，提取特征和标签
        for words, labels in zip(sentences, sentence_labels):
            previous_labels = ["O"] + labels
            tagged_words = pos_tag(words)  # 对句子进行词性标注
            features = [
                self.features(words, previous_labels[i], i, tagged_words=tagged_words)
                for i in range(len(words))
            ]
            all_words.extend(words)
            all_labels.extend(labels)
            all_features.extend(features)

        # 获取指定范围的特征
        (m, n) = bound
        pdists = self.classifier.prob_classify_many(all_features[m:n])

        print('  Words          P(PERSON)  P(O)\n' + '-' * 40)
        for (word, label, pdist) in zip(all_words[m:n], all_labels[m:n], pdists):
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

    def predict_sentence(self, text):
        """对输入的文本进行命名实体识别，支持多句子，并传递两种标签的概率"""
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
                pdist = self.classifier.prob_classify(feature)
                result = pdist.max()
                results.append(result)
                previous_labels.append(result)
                # 如果是命名实体，添加到结果中
                is_person = result == "PERSON"
                named_entities.append({
                    'word': words[i],
                    'is_person': is_person,
                    'prob_person': pdist.prob("PERSON"),
                    'prob_other': pdist.prob("O")
                })      
            # 打印当前句子的人名
            print(f"Sentence: {sentence}")
            print("Named Entities:", [entity['word'] for entity in named_entities if entity['is_person']])
            # 保存当前句子的命名实体结果
            all_named_entities.append(named_entities)
        return all_named_entities
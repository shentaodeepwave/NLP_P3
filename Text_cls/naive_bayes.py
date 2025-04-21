import json
import nltk
import argparse
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import os
from tqdm import tqdm
import math
import random
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(relative_path):
    """将相对路径转换为绝对路径"""
    return os.path.join(ROOT_DIR, relative_path)

import csv  # 添加 CSV 支持

def preprocess(inputfile, outputfile):
    """预处理 CSV 文件并输出为 JSON 格式"""
    nltk.download('punkt')
    stemmer = PorterStemmer()
    translator = str.maketrans('', '', string.punctuation)
    inputfile = get_absolute_path(inputfile)
    outputfile = get_absolute_path(outputfile)

    label_map = {'1': 'World', '2': 'Sports', '3': 'Business', '4': 'Sci/Tech'}  # 标签映射
    preprocessed_data = []

    with open(inputfile, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            label = label_map[row[0]]  # 将标签转换为类别名称
            text = row[2]  # 第三列为文本
            text = text.translate(translator)
            text = text.lower()
            tokens = word_tokenize(text)
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            preprocessed_data.append({
                'file_id': row[1],  # 第二列为文件 ID
                'category': label,
                'text': ' '.join(stemmed_tokens)
            })

    with open(outputfile, 'w', encoding='utf-8') as outfile:
        json.dump(preprocessed_data, outfile, indent=4)

def count_word(inputfile, outputfile):
    """统计单词频率并输出到文件"""
    inputfile = get_absolute_path(inputfile)
    outputfile = get_absolute_path(outputfile)

    with open(inputfile, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    word_counts = {}
    class_counts = {'World': 0, 'Sports': 0, 'Business': 0, 'Sci/Tech': 0}  # 类别名称

    for doc in data:
        category = doc['category']
        class_counts[category] += 1
        words = doc['text'].split()
        for word in words:
            if word not in word_counts:
                word_counts[word] = {'World': 0, 'Sports': 0, 'Business': 0, 'Sci/Tech': 0}
            word_counts[word][category] += 1

    with open(outputfile, 'w', encoding='utf-8') as outfile:
        outfile.write(' '.join(str(class_counts[cls]) for cls in ['World', 'Sports', 'Business', 'Sci/Tech']) + '\n')
        for word, counts in word_counts.items():
            outfile.write(f"{word} {' '.join(str(counts[cls]) for cls in ['World', 'Sports', 'Business', 'Sci/Tech'])}\n")

def feature_selection(inputfile,threshold,outputfile):
    #TODO: Choose the most frequent 10000 words(defined by threshold) as the feature word
    # Use the frequency obtained in 'word_count.txt' to calculate the total word frequency in each class.
    #   Notice that when calculating the word frequency, only words recognized as features are taken into consideration.
    # Output the result to the output file in the format required
    inputfile = get_absolute_path(inputfile)
    outputfile = get_absolute_path(outputfile)
    with open(inputfile, 'r') as infile:
        lines = infile.readlines()
    
    class_totals = list(map(int, lines[0].split()))
    word_counts = []

    for line in lines[1:]:
        parts = line.split()
        word = parts[0]
        counts = list(map(int, parts[1:]))
        total_count = sum(counts)
        word_counts.append((word, counts, total_count))
    
    word_counts.sort(key=lambda x: x[2], reverse=True)
    selected_words = word_counts[:threshold]

    with open(outputfile, 'w') as outfile:
        outfile.write(' '.join(map(str, class_totals)) + '\n')
        for word, counts, _ in selected_words:
            outfile.write(f"{word} {' '.join(map(str, counts))}\n")

def calculate_probability(word_count, word_dict, outputfile):
    word_count = get_absolute_path(word_count)
    word_dict = get_absolute_path(word_dict)
    outputfile = get_absolute_path(outputfile)

    with open(word_count, 'r') as wc_file, open(word_dict, 'r') as wd_file:
        wc_lines = wc_file.readlines()
        wd_lines = wd_file.readlines()

    class_totals = list(map(int, wd_lines[0].split()))
    total_docs = sum(class_totals)
    prior_probabilities = [count / total_docs for count in class_totals]

    vocabulary = set()
    for line in wd_lines[1:]:
        parts = line.split()
        vocabulary.add(parts[0])
    vocabulary_size = len(vocabulary)

    word_probabilities = {}
    for line in wd_lines[1:]:
        parts = line.split()
        word = parts[0]
        counts = list(map(int, parts[1:]))
        word_probabilities[word] = [
            (count + 1) / (class_totals[i] + vocabulary_size) 
            for i, count in enumerate(counts)
        ]


    with open(outputfile, 'w') as outfile:
        outfile.write(' '.join(map(str, prior_probabilities)) + '\n')
        for word, probabilities in word_probabilities.items():
            outfile.write(f"{word} {' '.join(map(str, probabilities))}\n")

def classify(probability, testset, outputfile):
    """实现朴素贝叶斯分类器"""
    probability = get_absolute_path(probability)
    testset = get_absolute_path(testset)
    outputfile = get_absolute_path(outputfile)

    # 读取概率文件
    with open(probability, 'r') as prob_file:
        prob_lines = prob_file.readlines()

    # 提取先验概率
    prior_probabilities = list(map(float, prob_lines[0].split()))

    # 提取单词的条件概率
    word_probabilities = {}
    for line in prob_lines[1:]:
        parts = line.split()
        word = parts[0]
        probabilities = list(map(float, parts[1:]))
        word_probabilities[word] = probabilities

    # 读取测试集
    test_data = []
    with open(testset, 'r', encoding='utf-8') as test_file:
        reader = csv.reader(test_file)
        for row in reader:
        
            file_id = row[1]  # 文件 ID
            text = row[2]  # 文本内容
            test_data.append((file_id, text))
            

    # 分类
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    results = []
    for file_id, text in test_data:
        tokens = word_tokenize(text.lower())
        scores = prior_probabilities.copy()  # 初始化为先验概率的对数值
        for i, category in enumerate(categories):
            for token in tokens:
                if token in word_probabilities:
                    scores[i] += math.log(word_probabilities[token][i])  # 使用对数避免下溢

        # 找到得分最高的类别
        predicted_category = categories[scores.index(max(scores))]
        results.append((file_id, predicted_category))

    # 写入分类结果
    with open(outputfile, 'w', encoding='utf-8') as outfile:
        for file_id, predicted_category in results:
            outfile.write(f"{file_id} {predicted_category}\n")


def f1_score(testset, classification_result):
    """计算 F1 分数"""
    testset = get_absolute_path(testset)
    classification_result = get_absolute_path(classification_result)
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    # 从测试集 CSV 文件读取真实标签
    true_labels = []
    with open(testset, 'r', encoding='utf-8') as test_file:
        reader = csv.reader(test_file)
        for row in reader:
            true_labels.append(categories[int(row[0]) - 1])  # 将标签转换为类别名称

    # 从分类结果文件读取预测标签
    predicted_labels = []
    with open(classification_result, 'r', encoding='utf-8') as result_file:
        for line in result_file:
            predicted_labels.append(line.strip().split(' ')[-1])  # 直接读取预测标签

    
    tp = {cls: 0 for cls in categories}
    fp = {cls: 0 for cls in categories}
    fn = {cls: 0 for cls in categories}

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label not in categories or (predicted_label and predicted_label not in categories):
            print(f"Warning: Skipping invalid label pair ({true_label}, {predicted_label})")
            continue
        if predicted_label == true_label:
            tp[true_label] += 1
        else:
            if predicted_label:
                fp[predicted_label] += 1
            fn[true_label] += 1

    f1_scores = []
    for cls in categories:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)

def accuracy_score(testset, classification_result):
    """计算分类模型的准确率"""
    testset = get_absolute_path(testset)
    classification_result = get_absolute_path(classification_result)
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    # 从测试集 CSV 文件读取真实标签
    true_labels = []
    with open(testset, 'r', encoding='utf-8') as test_file:
        reader = csv.reader(test_file)
        for row in reader:
            true_labels.append(categories[int(row[0]) - 1])  # 将标签转换为类别名称

    # 从分类结果文件读取预测标签
    predicted_labels = []
    with open(classification_result, 'r', encoding='utf-8') as result_file:
        for line in result_file:
            predicted_labels.append(line.strip().split(' ')[-1])  # 直接读取预测标签

    correct_predictions = sum(1 for true_label, predicted_label in zip(true_labels, predicted_labels) if true_label == predicted_label)
    total_predictions = len(true_labels)

    return correct_predictions / total_predictions if total_predictions > 0 else 0


def main():
    ''' Main Function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-pps', '--preprocess', type=str, nargs=2, help='preprocess the dataset')
    parser.add_argument('-cw', '--count_word', type=str, nargs=2, help='count the words from the corpus')
    parser.add_argument('-fs', '--feature_selection', type=str, nargs=3, help='select the features from the corpus')
    parser.add_argument('-cp', '--calculate_probability', type=str, nargs=3,
                        help='calculate the posterior probability of each feature word, and the prior probability of the class')
    parser.add_argument('-cl', '--classify', type=str, nargs=3,
                        help='classify the testset documents based on the probability calculated')
    parser.add_argument('-f1', '--f1_score', type=str, nargs=2,
                        help='calculate the F-1 score based on the classification result.')
    parser.add_argument('-acc', '--accuracy', type=str, nargs=2,
                        help='calculate the accuracy of the classification result.')

    opt = parser.parse_args()

    if opt.preprocess:
        input_file = opt.preprocess[0]
        output_file = opt.preprocess[1]
        preprocess(input_file, output_file)
    elif opt.count_word:
        input_file = opt.count_word[0]
        output_file = opt.count_word[1]
        count_word(input_file, output_file)
    elif opt.feature_selection:
        input_file = opt.feature_selection[0]
        threshold = int(opt.feature_selection[1])
        outputfile = opt.feature_selection[2]
        feature_selection(input_file, threshold, outputfile)
    elif opt.calculate_probability:
        word_count = opt.calculate_probability[0]
        word_dict = opt.calculate_probability[1]
        output_file = opt.calculate_probability[2]
        calculate_probability(word_count, word_dict, output_file)
    elif opt.classify:
        probability = opt.classify[0]
        testset = opt.classify[1]
        outputfile = opt.classify[2]
        classify(probability, testset, outputfile)
    elif opt.f1_score:
        testset = opt.f1_score[0]
        classification_result = opt.f1_score[1]
        f1 = f1_score(testset, classification_result)
        print('The F1 score of the classification result is: ' + str(f1))
    elif opt.accuracy:
        testset = opt.accuracy[0]
        classification_result = opt.accuracy[1]
        accuracy = accuracy_score(testset, classification_result)
        print('The accuracy of the classification result is: ' + str(accuracy))

def preprocess_text():
    stemmer = PorterStemmer()
    translator = str.maketrans('', '', string.punctuation)
    preprocessed_data = []
    stemmer = PorterStemmer()
    translator = str.maketrans('', '', string.punctuation)

    with open('./Text_cls/test.csv', 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        random_texts = random.sample(list(reader), 5)  # 随机抽取五条文本
    label_map = {'1': 'World', '2': 'Sports', '3': 'Business', '4': 'Sci/Tech'}
    for row in random_texts:
        label = label_map[row[0]]  # 将标签转换为类别名称
        text = row[2]  # 第三列为文本
        text = text.translate(translator)
        text = text.lower()
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        preprocessed_data.append({
            'file_id': row[1],  # 第二列为文件 ID
            'category': label,
            'text': row[2],  # 第三列为文本
            'preprocess_text': ' '.join(stemmed_tokens)  # 添加预处理后的文本
        })
    text = [record['text'] for record in preprocessed_data]
    preprocess_text = [record['preprocess_text'] for record in preprocessed_data]  # 提取预处理后的文本
    return text, preprocess_text,label

def classify1(probability, input_text):
    """实现朴素贝叶斯分类器"""
    with open(probability, 'r') as prob_file:
        prob_lines = prob_file.readlines()

    prior_probabilities = list(map(float, prob_lines[0].split()))
    word_probabilities = {}
    for line in prob_lines[1:]:
        parts = line.split()
        word = parts[0]
        probabilities = list(map(float, parts[1:]))
        word_probabilities[word] = probabilities

    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    tokens = word_tokenize(input_text.lower())
    scores = prior_probabilities.copy()
    for i, category in enumerate(categories):
        for token in tokens:
            if token in word_probabilities:
                scores[i] += math.log(word_probabilities[token][i])  # 使用对数避免下溢

    predicted_category = categories[scores.index(max(scores))]
    return predicted_category, scores  # 返回分类结果和得分

if __name__ == '__main__':
    import os
    main()
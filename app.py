from flask import Flask, request, render_template
from NER.MEM4 import MEMM
import random
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
app = Flask(__name__)
classifier = MEMM()
nltk.download('punkt')
import csv
# 加载文本分类原数据
def preprocess_text():
    stemmer = PorterStemmer()
    translator = str.maketrans('', '', string.punctuation)
    preprocessed_data = []
    stemmer = PorterStemmer()
    translator = str.maketrans('', '', string.punctuation)

    with open('./Text-cls/test.csv', 'r', encoding='utf-8') as infile:
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
import math  # 添加 math 模块

# 添加分类功能
import math  # 添加 math 模块




@app.route('/', methods=['GET'])
def index():

    text,preprocess,label = preprocess_text()
    #只取文本
    
    return render_template('index.html', named_entities=None, random_texts=text, classification_result=None)

def classify(probability, input_text):
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


@app.route('/predict', methods=['POST'])
def predict():
    try:
        classifier.load_model()
        sentence = request.form.get('sentence')
        text, preprocess, label = preprocess_text()
        if not sentence:
            return render_template('index.html', error="输入句子为空", named_entities=None, random_texts=text, classification_result=None)

        probability_file = './Text-cls/word_probability.txt'
        classification_result, scores = classify(probability_file, sentence)

        categories = ['World', 'Sports', 'Business', 'Sci/Tech']
        named_entities = classifier.predict_sentence(sentence)

        return render_template(
            'index.html',
            named_entities=named_entities,
            error=None,
            random_texts=text,
            classification_result=[f"分类结果: {classification_result}"],
            scores=scores,
            categories=categories
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        text, preprocess, label = preprocess_text()
        return render_template('index.html', error=f"Error: {str(e)}", named_entities=None, random_texts=text, classification_result=None)

if __name__ == '__main__':
    app.run(debug=True)
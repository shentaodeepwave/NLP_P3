from flask import Flask, request, render_template
from NER.MEM3 import MEMM
import random
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
app = Flask(__name__)
classifier = MEMM()
nltk.download('punkt')

# 加载文本分类原数据
def preprocess_text():
    with open('./Text-cls/train.json', 'r', encoding='utf-8') as f:
        raw_texts = json.load(f)
    stemmer = PorterStemmer()
    translator = str.maketrans('', '', string.punctuation)
    random_texts = random.sample(raw_texts, 5)  # 随机抽取五条文本
    data=random_texts
    preprocessed_data = []

    for record in data:
        file_id = record[0]
        category = record[1]
        text = record[2]
        #根据\n分割文本
        text = text.split('\n')
        #去掉空行
        text = [line for line in text if line.strip() != '']
        #拼接文本
        text = ' '.join(text)
        tokens = word_tokenize(text)
        preprocessed_data.append({
            'file_id': file_id,
            'category': category,
            'text': ' '.join(tokens)
        })
    preprocessed_data = [record['text'] for record in preprocessed_data]
    return preprocessed_data, random_texts

@app.route('/', methods=['GET'])
def index():

    preprocess_data, raw_texts = preprocess_text()
    #只取文本
    
    return render_template('index.html', named_entities=None, random_texts=preprocess_data, classification_result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        classifier.load_model()
        sentence = request.form.get('sentence')
        preprocess_data, raw_texts = preprocess_text()
        if not sentence:
            return render_template('index.html', error="输入句子为空", named_entities=None, random_texts=preprocess_data, classification_result=None)
        
        # 命名实体识别
        named_entities = classifier.predict_sentence(sentence)

        # 文本分类逻辑（假设分类结果为随机生成）
        
        classification_result = [f"分类结果: {random.choice(['crude', 'grain', 'money-fx', 'acq', 'earn'])}"]
        
        preprocess_data, raw_texts = preprocess_text()
        return render_template('index.html', named_entities=named_entities, error=None, random_texts=preprocess_data, classification_result=classification_result)
    except Exception as e:
        print(f"Error occurred: {e}")
        preprocess_data, raw_texts = preprocess_text()
        return render_template('index.html', error=f"Error: {str(e)}", named_entities=None, random_texts=preprocess_data, classification_result=None)

if __name__ == '__main__':
    app.run(debug=True)
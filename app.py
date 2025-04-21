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
from Text_cls.naive_bayes import preprocess_text,classify1



@app.route('/', methods=['GET'])
def index():

    text,preprocess,label = preprocess_text()
    #只取文本
    
    return render_template('index.html', named_entities=None, random_texts=text, classification_result=None)




@app.route('/predict', methods=['POST'])
def predict():
    try:
        classifier.load_model()
        sentence = request.form.get('sentence')
        text, preprocess, label = preprocess_text()
        if not sentence:
            return render_template('index.html', error="输入句子为空", named_entities=None, random_texts=text, classification_result=None)

        probability_file = './Text_cls/word_probability.txt'
        classification_result, scores = classify1(probability_file, sentence)

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
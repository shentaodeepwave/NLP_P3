from flask import Flask, request, render_template
from NER.MEM3 import MEMM

app = Flask(__name__)
classifier = MEMM()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', named_entities=None)  # 确保初始页面渲染时 named_entities 为 None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        classifier.load_model()
        sentence = request.form.get('sentence')
        if not sentence:
            return render_template('index.html', error="输入句子为空", named_entities=None)
        named_entities = classifier.predict_sentence(sentence)
        print(f"Named Entities1111: {named_entities}")
        return render_template('index.html', named_entities=named_entities, error=None)
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('index.html', error=f"Error: {str(e)}", named_entities=None)

if __name__ == '__main__':
    app.run(debug=True)
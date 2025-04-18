from flask import Flask, request, render_template
from MEM3 import MEMM

app = Flask(__name__)
classifier = MEMM()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        classifier.load_model()
        sentence = request.form.get('sentence')
        named_entities = classifier.predict_sentence(sentence)
        return render_template('index.html', named_entities=named_entities)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
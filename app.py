from flask import Flask, render_template, request
from src.module import Predictor

app = Flask(__name__)
predictor = Predictor()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    result = predictor.predict_class(message)

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
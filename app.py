from flask import Flask,render_template,request
import numpy as np
import CKDprediction as ckd
app = Flask(__name__)
import json
data = []
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        attributes = ['white blood cell count', 'blood urea', 'serum creatinine', 'blood glucose random', 'pedal edema', 'pus cell', 'sugar', 'appetite', 'albumin',  'hypertension']
        values = []
        for attribute in attributes:
            value = request.form.get(attribute)
            values.append(float(value))
        print(values)
        import CKDprediction as ckd
        
        p = ckd.predict(values)
        v = p.printPrediction()
        return render_template('input.html',v=v)
        
    else:
        return render_template('input.html')

if __name__ == '__main__':
    app.run(debug=True)
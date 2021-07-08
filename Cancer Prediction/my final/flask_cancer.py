import os
mycwd = 'C:\\Users\\ASUS\\Desktop\\sharikhh\\Data Science\\Machine Learning 2\\Project\\Final File\\my final'
os.chdir(mycwd)
os.getcwd()
#os.listdir()
# an object of WSGI application
#from flask import Flask	
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__) # Flask constructor

variety_mappings = {2: 'Benign', 4: 'Malignant'}



#app = Flask(__name__)
logreg = pickle.load(open('Final_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('demo_logreg.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_features = [float(x) for x in request.form.values()]
    final_features = np.array(input_features)
    query = final_features.reshape(1,-1)
    output = variety_mappings[logreg.predict(query)[0]]
    return render_template('demo_logreg.html', prediction_text='Variety is:{}'.format(output))


if __name__ == "__main__":
    app.run(debug=False)

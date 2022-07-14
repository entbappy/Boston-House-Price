'''
@Author: Bappy Ahmed
Date: 03 sep 2021
Email: entbappy73@gmail.com
'''


from utils.outliers import RemoveOutliers
import pickle5 as pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

pipe = pickle.load(open('Model/pipe.pkl', 'rb'))


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            CRIM =float(request.form['CRIM'])
            ZN =float(request.form['ZN'])
            INDUS =float(request.form['INDUS'])
            CHAS =request.form['CHAS']
            NOX = float(request.form['NOX'])
            RM = float(request.form['RM'])
            AGE = float(request.form['AGE'])
            DIS = float(request.form['DIS'])
            TAX = float(request.form['TAX'])
            PTRATIO = float(request.form['PTRATIO'])
            B = float(request.form['B'])
            LSTAT = float(request.form['LSTAT'])


            outliers = RemoveOutliers()
            columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
            data = [CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,TAX,PTRATIO,B,LSTAT]
            data = np.array(data).reshape(1, 12)
            data = pd.DataFrame(data, columns=columns)

            for col in data.columns:
                ignore = ['CHAS']
                if col not in ignore:
                    data[col] = data[col].astype(float)

            out = outliers.remove(data)
            output = pipe.predict(out)[0]
            print(output)

            return render_template('results.html', prediction = str(np.round(output* 1000,4)))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')



if __name__ == "__main__":
	app.run(debug=True)

# # Load the Model
import pickle
from flask import Flask
from flask import request
from flask import jsonify

app = Flask('predict')
model_file = 'model_C=1.bin'

with open(model_file, 'rb') as f_in: # 'rb' because we are reading into binary file.
    dv, model = pickle.load(f_in)


@app.route('/predict', methods = ['POST'])
def predict():
    customer = request.get_json()
    customer_dict = dv.transform([customer])
    y_pred = model.predict_proba(customer_dict)[0, 1]
    churn = y_pred >= 0.55

    result = {
        'churn probability' : float(y_pred),
        'churn':bool(churn)
    } 
    # If we didn't convert these result variables to python float and bool we would get response [500].
    #This conflict arises due to numpy bool and float, so therefore we convert it to python variables.

    return jsonify(result)


# This if statement is not executed when this file is run through gunicorn.
# gunicorn is a production server.
# Look for the difference between deployment and production server.
if __name__ == '__main__':
    app.run(debug= True, host= '0.0.0.0', port= 9696)
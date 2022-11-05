
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import joblib
import traceback
from sklearn.metrics import mean_squared_error
import os
from flask_restx import Api, Resource

class modelClass:
    def __init__(self, input_model, hyperparam1, hyperparam2):

        '''initialize class
        params:
        input_model : str, name of model (Ridge or RandomForestRegressor)
        hyperparam1 = int, hyperparam 1 for model
        hyperparam2 = int, hyperparam 2 for model
        '''

        X_raw, self.y = load_boston(return_X_y=True)
        self.X = pd.DataFrame(X_raw)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=10)

        if input_model == 'Ridge':
            self.model_name = 'Ridge'
            self.model = Ridge(alpha = hyperparam1, max_iter = hyperparam2)
        else:
            self.model_name = 'RandomForestRegressor'
            self.model=RandomForestRegressor(max_depth = hyperparam1, min_samples_leaf = hyperparam2)

    def fit(self):
        '''
        fit model
        '''
        self.model.fit(self.X_train, self.y_train)

    def save_model(self):
        '''
        Serialize model and its columns and save 
        '''
        joblib.dump(self.model, self.model_name +'.pkl')
        print("Model is saved")
        rnd_columns = list(self.X.columns)
        joblib.dump(rnd_columns, self.model_name +'_columns.pkl')
        print("Model Colums are Saved")

    def load_model(self):
        '''
        Load the model
        '''
        model = joblib.load(self.model_name +'.pkl')
        print(model)
        print("Model is loaded")
        return model

    def predict(self, x_test):
        '''
        Make prediction
        params:
        x_test : data frame of features for prediction
        '''
        model = self.load_model()
        y_pred = model.predict(x_test)
        return y_pred


app = Flask(__name__)
application = Api(app)

@application.route("/getListOfModels")
class getListOfModels(Resource):
    @application.response(200, 'OK')
    @application.response(400, 'BAD REQUEST')

    def get(self):
        '''
        API to get list of available models
        no params
        '''
        print("getListOfModels")
        return jsonify({"Model 1": 'Ridge', "Model 2": 'RandomForestRegressor'})

@application.route("/learnDefaultModel")
class learnDefaultModel(Resource):
    @application.response(200, 'OK')
    @application.response(400, 'BAD REQUEST')

    def post(self):
        '''
        API to learn default model and get prediction
        params:
        json: {"model":"Ridge", "0":0.1, "1":100}
        '''
        print("learnDefaultModel")
        json_ = request.get_json()
        if json_[0]['model'] not in ['Ridge', 'RandomForestRegressor']:
            return "Model is not available", 400
        print(json_[0]['model'])

        model = modelClass(json_[0]['model'], json_[0]['0'], json_[0]['1'])

        model.fit()
        model.save_model()
        pred= model.predict(model.X_test)

        return jsonify({"predicion": list(pred)})


@application.route("/predictCustomData")
class predictOnCustomData(Resource):
    @application.response(200, 'OK')
    @application.response(400, 'BAD REQUEST')

    def post(self):
        '''
        API to get predicion on user's custom data model
        params:
        json: [{"model":"Ridge", "0":0.1,"1":100},
                {"0":0.05479,"1":33.0,"2":2.18,"3":0.0,"4":0.472,
                "5":6.616,"6":58.1,"7":3.3700, "8":7.0,"9":222.0,
                "10":18.4,"11":393.36,"12":8.93}]
        '''
        print("predictOnCustomData")
        json_ = request.get_json()
        if json_[0]['model'] not in ['Ridge', 'RandomForestRegressor']:
            return "Model is not available", 400
        print(json_[0]['model'])

        model = modelClass(json_[0]['model'], json_[0]['0'], json_[0]['1'])

        model.fit()
        model.save_model()

        X_test = json_[1]
        
        clmns = joblib.load(model.model_name +'_columns.pkl')
        X_test = list(X_test.values())

        X_test = pd.DataFrame([X_test], columns=clmns) 

        model.fit()
        model.save_model()
        pred = model.predict(X_test)

        return jsonify({"predicion": list(pred)})

@application.route("/deleteModel")
class deleteModel(Resource):
    @application.response(200, 'OK')
    @application.response(400, 'BAD REQUEST')

    def delete(min_samples_leaf):
        '''
        API to delete given model
        params:
        json: [{"model":"Ridge"}]
        '''
        print("deleteModel")
        json_ = request.get_json()
        if json_[0]['model'] not in ['Ridge', 'RandomForestRegressor']:
            return "Model is not available",400
        else:
            try:
                res = os.remove("./"+json_[0]['model']+'.pkl')
                return str('Model ' + json_[0]['model'] + " is removed"),200
            except:
                return "Model is not found",400


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except:
        port = 12345

app.run(port=port, debug=True)
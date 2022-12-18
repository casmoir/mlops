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
from flask_restx import Api, Resource, reqparse
from modelClass import modelClass
import json

app = Flask(__name__)
application = Api(app)

learnDefaultModel_param = reqparse.RequestParser()
learnDefaultModel_param.add_argument('model_id', help='Unique model id')
learnDefaultModel_param.add_argument('model', help='name of the model', choices=['RandomForestRegressor', 'Ridge'])
learnDefaultModel_param.add_argument('params', help='dict with model parameters', type = dict)


@application.route("/getListOfModels", doc={'description': 'Get list of available models, no params'})
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

@application.route("/learnDefaultModel", doc={'description': 'Learn default model and get prediction'})
class learnDefaultModel(Resource):
    @application.response(200, 'OK')
    @application.response(400, 'BAD REQUEST')
    @application.response(500, 'BAD PARAMETER')
    @application.expect(learnDefaultModel_param)

    def post(self):
        '''
        API to learn default model and get prediction
        params:
        json: {"model":"Ridge", "0":0.1, "1":100}
        '''
        print("learnDefaultModel")

        rgs = learnDefaultModel_param.parse_args()
        print(rgs)
        if rgs['model'] not in ['Ridge', 'RandomForestRegressor']:
            return "Model is not available", 400
        rgs.params = dict(rgs.params)

        model = modelClass(rgs['model'], rgs.params)

        model.fit()
        model.save_model(rgs['model_id'])
        pred= model.predict(model.X_test)

        return jsonify({"predicion": list(pred)})


@application.route("/predictCustomData", doc={'description': 'Use default model and get prediction on custom data'})
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

@application.route("/deleteModel", doc={'description': 'Delete model'})
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
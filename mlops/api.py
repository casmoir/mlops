#from sklearn.datasets import load_boston
#from sklearn.linear_model import Ridge
import pandas as pd
#rom sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
from flask import Flask, jsonify
import joblib
#import traceback
#from sklearn.metrics import mean_squared_error
import os
from flask_restx import Api, Resource, fields
from modelClass import modelClass
#import json


app = Flask(__name__)
application = Api(app)

learnDefaultModel_param = application.model('learn default model',
                             {'model_id': fields.Integer(description='Model id', example='0'),
                              'model': fields.String(description='name of the model', example='Ridge'),
                              'params': fields.Arbitrary(description='dict with model parameters', example= {'alpha': 0.1, 'max_iter': 100})})

predictOnCustomData_param = application.model("get predicion on user's custom data",
                              {'model_name': fields.String(description='Name of existing model with id', example='0Ridge'),
                              'params': fields.Arbitrary(description='dict with model features for prediction', example= {"0":0.05479,"1":33.0,"2":2.18,"3":0.0,"4":0.472,
                "5":6.616,"6":58.1,"7":3.3700, "8":7.0,"9":222.0,
                "10":18.4,"11":393.36,"12":8.93})})

deleteModel_param = application.model('delete model',
                             {'model': fields.String(description='name of the model with id', example='Ridge')})


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
@application.doc(params={'model_id': 'id of model', 'model': 'name of model', 'params' : 'hyperparameters for model'})
class learnDefaultModel(Resource):
    @application.response(200, 'OK')
    @application.response(400, 'BAD REQUEST')
    @application.response(500, 'BAD PARAMETER')
    @application.expect(learnDefaultModel_param)

    def post(self):
        '''
        API to learn default model and get prediction
        params:
        json: {'model_id': 'default', 'model': 'Ridge', 'params': {'alpha': 0.1, 'max_iter': 100}}
        '''
        print("learnDefaultModel")

        model_id = application.payload['model_id']
        model_name= application.payload['model']
        data = (application.payload['params'])

        if model_name not in ['Ridge', 'RandomForestRegressor']:
            return "Model is not available", 400

        model = modelClass(model_name, data)

        model.fit()
        model.save_model(model_id)
        pred= model.predict(model.X_test)

        return jsonify({"predicion": list(pred)})


@application.route("/predictCustomData", doc={'description': 'Use default model and get prediction on custom data'})
@application.doc(params={'model_name': 'name of model with id', 'params': 'features to get prediction'})
class predictOnCustomData(Resource):
    @application.response(200, 'OK')
    @application.response(400, 'BAD REQUEST')
    @application.expect(predictOnCustomData_param)

    def post(self):
        '''
        API to get predicion on user's custom data model
        params:
        json: {"model_name":"0Ridge"}, 
                "params":{"0":0.05479,"1":33.0,"2":2.18,"3":0.0,"4":0.472,
                "5":6.616,"6":58.1,"7":3.3700, "8":7.0,"9":222.0,
                "10":18.4,"11":393.36,"12":8.93}
        '''
        print("predictOnCustomData")

  
        model_name= application.payload['model_name']
        data = (application.payload['params'])

        X_test = data
        
        try:
            clmns = joblib.load('saved_models/' + model_name +'_columns.pkl')
        except:
            return "Model is not not available", 400
        X_test = list(X_test.values())

        X_test = pd.DataFrame([X_test], columns=clmns) 

        model = joblib.load('saved_models/'+ model_name +".pkl") 

        pred = model.predict(X_test)

        return jsonify({"predicion": list(pred)})

@application.route("/deleteModel", doc={'description': 'Delete model'})
@application.doc(params={'model': 'name of model with id'})
class deleteModel(Resource):
    @application.response(200, 'OK')
    @application.response(400, 'BAD REQUEST')
    @application.expect(deleteModel_param)

    def delete(min_samples_leaf):
        '''
        API to delete given model
        params:
        json: {"model":"0Ridge"}
        '''
        print("deleteModel")

        model_name= application.payload['model']

        try:
            res = os.remove('saved_models/'+ model_name +".pkl") 
            return str('Model ' + model_name + " is removed"),200
        except:
            return "Model is not found",400
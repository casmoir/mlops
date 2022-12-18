import pandas as pd
import joblib
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import joblib
import traceback
from sklearn.metrics import mean_squared_error
import psycopg2
from sqlalchemy import create_engine


def get_data():
    engine = create_engine('postgresql://postgres:password@localhost:5432/mlops')
    df = pd.read_sql_query('select * from "boston"',con=engine)
    df = df.iloc[: , 1:]
    y = df['target']
    X = df.drop('target', axis = 1)
    print(X)
    return X,y

#get_data()

class modelClass:
    def __init__(self, input_model, model_params):

        '''initialize class
        params:
        input_model : str, name of model (Ridge or RandomForestRegressor)
        hyperparam1 = int, hyperparam 1 for model
        hyperparam2 = int, hyperparam 2 for model
        '''
        #data_url = '/Users/anastasiaraeva/mlops/mlops/mlops/data/boston.csv'
        #raw_df = pd.DataFrame(pd.read_csv(data_url, sep = ';'))
        #raw_df = raw_df.iloc[: , 1:]
        #print(raw_df)
        #self.y = raw_df['target']
        #self.X = raw_df.drop('target', axis = 1)
        self.X, self.y = get_data()

        #X_raw, self.y = load_boston(return_X_y=True)
        #self.X = pd.DataFrame(X_raw)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=10)


        models = {'RandomForestRegressor': RandomForestRegressor(), 'Ridge': Ridge()}

        if input_model == 'Ridge':
            self.model_name = 'Ridge'
            #self.model = Ridge(alpha = hyperparam1, max_iter = hyperparam2)
        else:
            self.model_name = 'RandomForestRegressor'
            #self.model=RandomForestRegressor(max_depth = hyperparam1, min_samples_leaf = hyperparam2)

        for param in model_params:
            if param not in models[self.model_name].get_params().keys():
                return "Invalid model parameter", 500
        self.model = models[self.model_name].set_params(**model_params) 

    def fit(self):
        '''
        fit model
        '''
        self.model.fit(self.X_train, self.y_train)

    def save_model(self, name = 'default'):
        '''
        Serialize model and its columns and save 
        '''
        joblib.dump(self.model,'saved_models/'+str(name)+ self.model_name +'.pkl')
        print("Model is saved")
        rnd_columns = list(self.X.columns)
        joblib.dump(rnd_columns, 'saved_models/'+str(name) + self.model_name +'_columns.pkl')
        print("Model Colums are Saved")

    def load_model(self, name = 'default', mode = ''):
        '''
        Load the model
        '''
        model = joblib.load('saved_models/'+str(name) + self.model_name + mode + '.pkl')
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
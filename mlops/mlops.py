#from sklearn.model_selection import train_test_split
#from flask import Flask
#import joblib
#import traceback
#from sklearn.metrics import mean_squared_error
#import os
#from flask_restx import Api, Resource
#from modelClass import modelClass
from api import *

#app = Flask(__name__)
#application = Api(app)

if __name__ == "__main__":
    #try:
     #   port = int(sys.argv[1])
    #except:
     #   port = 5000
        
    app.run(port=12345, debug=True)
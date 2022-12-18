
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
#import joblib
#import traceback
from sklearn.metrics import mean_squared_error
#import os
from flask_restx import Api, Resource
from modelClass import modelClass
from api import *


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
        
    app.run(port=port, debug=True)
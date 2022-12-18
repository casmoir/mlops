from api import getListOfModels
import pandas as pd
from flask import Flask

app = Flask(__name__)

with app.test_client() as c:
    rv = c.post('/getListOfModels')
    json_data = rv.get_json()
    print(json_data)
    #assert verify_token(email, json_data['token'])

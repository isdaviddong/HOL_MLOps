#import os
import logging
import json
import numpy
import joblib
from sklearn.ensemble import RandomForestClassifier
from azureml.core.model import Model

def init():
    global rmodel

    # load the model from file into a global object
    try:
        logging.basicConfig(level=logging.DEBUG)
        #AutoML299e3f00319 要換成你的AutlMO Model名稱
        model_path = Model.get_model_path(model_name='AutoML99c02cfb423')
        logging.info(f"model_path ----> {model_path}")
        #載入model到rmodel全域物件
        rmodel = joblib.load(model_path)
        logging.info(f"rmodel ----> {rmodel}")
    except Exception as e:
        print(str(e))

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        logging.info(f"json.loads:{data}")
        return json.dumps({"echo data": data})
    except Exception as e:
        err = str(e)
        return f"ERROR : {err} \n data : {raw_data}  \n model : {rmodel}"
 
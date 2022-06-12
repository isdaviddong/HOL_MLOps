import os
import sys

import azureml as aml
from azureml.core import Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.core.run import Run
import argparse
import json
import time

import logging
import numpy as np

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset
from azureml.train.automl import AutoMLConfig

from azureml.core.run import Run
# workspace from config file
ws = Workspace.from_config()

#set-up Experiment Name
experiment = Experiment(workspace=ws, name="TestAutoMLExp")

#data source(from github)
data = "https://raw.githubusercontent.com/MicrosoftLearning/mslearn-dp100/main/data/diabetes.csv"
dataset = Dataset.Tabular.from_delimited_files(data)
training_data, validation_data = dataset.random_split(percentage=0.7, seed=223)
label_column_name = "Diabetic"

automl_settings = {
    "n_cross_validations": 3,
    "primary_metric": "AUC_weighted",
    "experiment_timeout_hours": 0.25,  # This is a time limit for testing purposes, remove it for real use cases, this will drastically limit ability to find the best model possible
    "verbosity": logging.INFO,
    "enable_stack_ensemble": False,
}

automl_config = AutoMLConfig(
    task="classification",
    debug_log="automl_errors.log",
    training_data=training_data,
    label_column_name=label_column_name,
    **automl_settings,
)

#local_run
local_run = experiment.submit(automl_config, show_output=True)

#get best_model
best_run, best_model = local_run.get_output()

#show model_name
model_name = best_run.properties['model_name']
print("model_name:" + model_name)
description = 'AutoML forecast example'
tags = None

#register_model
model = local_run.register_model(model_name = model_name,description = description,tags = tags)

#show model id for deploy
print(model.id)
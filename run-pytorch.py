# run-pytorch.py
import sys
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

if __name__ == "__main__":

    #show from_config
    file = open("config.json")
    line = file.read().replace("\n", " ")
    file.close()
    print(line)
    
    #get workspace from_config
    ws =  Workspace.from_config()
    experiment = Experiment(workspace=ws, name='day1-experiment-train')
    config = ScriptRunConfig(source_directory='./src',
                             script='train.py',
                             compute_target='cpu-cluster2')

    # use curated pytorch environment 
    #env = ws.environments['AzureML-PyTorch-1.6-CPU']
    env_name = 'AzureML-PyTorch-1.6-CPU'
    env = Environment.get(workspace=ws, name=env_name)
    config.run_config.environment = env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url) 
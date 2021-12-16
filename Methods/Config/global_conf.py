#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import sys
import socket
import os

hostname = socket.gethostname()
global_params = lambda: 0
global_params.cluster = \
(hostname[:5]=='local') * 'local' + \
(hostname[:2]=='eu') * 'euler' + \
(hostname[:5]=='daint') * 'daint' + \
(hostname[:14]=='barrycontainer') * 'barry' + \
(hostname[:3]=='nid') * 'daint'

HIPPO = False
if global_params.cluster == 'euler':
    print("## CONFIG: RUNNING IN EULER CLUSTER.")
    SCRATCH = os.environ['SCRATCH']
    project_path = SCRATCH + "/LED-Molecular/Code"

elif global_params.cluster == 'barry':
    print("## CONFIG: RUNNING IN BARRY CLUSTER.")
    # SCRATCH = os.environ['SCRATCH']
    SCRATCH = "/scratch/pvlachas"
    project_path = SCRATCH + "/LED-Molecular/Code"

elif global_params.cluster == 'daint':
    print("## CONFIG: RUNNING IN DAINT CLUSTER.")
    SCRATCH = os.environ['SCRATCH']
    project_path = SCRATCH + "/LED-Molecular/Code"

    # PROJECTFOLDER = "/project/s929/pvlachas"
    # project_path = PROJECTFOLDER + "/LED-Molecular/Code"

elif global_params.cluster == 'local':
    # Running in the local repository, pick whether you are loading data-sets from the hippo database, or using a local data folder.
    if HIPPO:
        print("## CONFIG: DATA LOADING FROM HIPPO.")
        HOME = os.environ['HOME']
        project_path = HOME + "/hippo/LED-Molecular/Code"
    else:
        print("## CONFIG: RUNNING IN LOCAL REPOSITORY.")
        config_path = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(os.path.dirname(config_path))

else:
    # IF NOTHING FROM THE ABOVE, RESORT TO LOCAL
    print(
        "## CONFIG: RUNNING IN LOCAL REPOSITORY (hostname {:} not resolved).".
        format(hostname))
    # raise ValueError("Avoid running in local repository.")
    if HIPPO:
        print("## CONFIG: DATA LOADING FROM HIPPO.")
        HOME = os.environ['HOME']
        project_path = HOME + "/hippo/LED-Molecular/Code"
    else:
        print("## CONFIG: RUNNING IN LOCAL REPOSITORY.")
        config_path = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(os.path.dirname(config_path))

print("PROJECT PATH={}".format(project_path))

global_params.global_utils_path = "./Models/Utils"

global_params.saving_path = project_path + "/Results/{:s}"
global_params.project_path = project_path

global_params.data_path_train = project_path + "/Data/{:s}/Data/train"
global_params.data_path_val = project_path + "/Data/{:s}/Data/val"
global_params.data_path_test = project_path + "/Data/{:s}/Data/test"
global_params.data_path_gen = project_path + "/Data/{:s}/Data"

# PATH TO LOAD THE PYTHON MODELS
global_params.py_models_path = "./Models/{:}"

# PATHS FOR SAVING RESULTS OF THE RUN
global_params.model_dir = "/Trained_Models/"
global_params.fig_dir = "/Figures/"
global_params.results_dir = "/Evaluation_Data/"
global_params.logfile_dir = "/Logfiles/"

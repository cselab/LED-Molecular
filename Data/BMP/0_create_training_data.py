#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
import pickle
import os
import h5py

def writeToHDF5Files(sequences, data_dir_str):
    data_dir = "./Data/{:}".format(data_dir_str)
    os.makedirs(data_dir, exist_ok=True)
    hf = h5py.File(data_dir + '/data.h5', 'w')
    # Only a single sequence_example per dataset group
    for seq_num_ in range(np.shape(sequences)[0]):
        print('batch_{:010d}'.format(seq_num_))
        data_group = sequences[seq_num_]
        data_group = np.array(data_group)
        # print(np.shape(data_group))
        gg = hf.create_group('batch_{:010d}'.format(seq_num_))
        gg.create_dataset('data', data=data_group)
    hf.close()


datadir = "./Plots"
figdir = "./Data"

os.makedirs(datadir, exist_ok=True)
os.makedirs(figdir, exist_ok=True)


trajectories_all = []
# N_RUNS = 100
# N_RUNS = 96
N_RUNS = 10
for run in range(N_RUNS):
    with open("./Simulation_Data/trajectories_ic_{:}.pickle".format(run), "rb") as file:
        data = pickle.load(file)
        time_per_iter = data["time_per_iter"]
        trajectories = data["trajectories"]
        dt_all = data["dt"]
        N_particles = data["N_particles"]
        trajectories_all.append(trajectories[0])

trajectories_all = np.array(trajectories_all)
N_init_transients = 100000
trajectories_all = trajectories_all[:, N_init_transients:]

SUBSAMPLE=50
trajectories = trajectories_all[:, ::SUBSAMPLE] 
dt = SUBSAMPLE * dt_all
print("dt={:}".format(dt))

print(np.shape(trajectories))
# (96, 20000, 2)

min_ = np.min(trajectories, axis=(0,1))
max_ = np.max(trajectories, axis=(0,1))

print("min = {:}".format(min_))
print("max = {:}".format(max_))

print(ark)

# 999999/4001 = 248
timestep_per_sequence       = 12001
num_ics_in_dataset_train    = 32
num_ics_in_dataset_val      = 32
num_ics_in_dataset_test     = 96


data                        = trajectories[:num_ics_in_dataset_train, :timestep_per_sequence]
writeToHDF5Files(data, "train")

data                        = trajectories[num_ics_in_dataset_train:num_ics_in_dataset_train+num_ics_in_dataset_val, :timestep_per_sequence]
writeToHDF5Files(data, "val")

data                        = trajectories[:num_ics_in_dataset_test, :timestep_per_sequence]
writeToHDF5Files(data, "test")




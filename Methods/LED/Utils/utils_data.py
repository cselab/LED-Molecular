#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

##############################################################
### SAVING AND LOADING FILES PROTOCOLS
##############################################################
import pickle
import hickle
import time
from . import utils_time

import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        scaler: Scaler transform to apply to every data instance (default=None).
    """
    def __init__(self,
                 file_path,
                 recursive,
                 load_data,
                 data_cache_size=3,
                 data_info_dict=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size

        if data_info_dict is not None:
            if "scaler" in data_info_dict.keys():
                self.scaler = data_info_dict["scaler"]
            else:
                self.scaler = None

            if "truncate" in data_info_dict:
                self.truncate = data_info_dict["truncate"]
            else:
                self.truncate = None
        else:
            self.truncate = None
            self.scaler = None

        # Search for all h5 files
        p = Path(file_path)
        assert (p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')
        # print(files)
        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.truncate is not None:
            x = x[:self.truncate[0], :self.truncate[1]]
        if self.scaler is not None:
            x = self.scaler.scaleData(x, single_sequence=True)
        return x

    def __len__(self):
        return len(self.get_data_infos('data'))

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path, 'r') as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                # print(gname)
                # print(group)
                # print(ark)
                for dname, ds in group.items():
                    # print(dname)
                    # print(ds)
                    # print(ds.value)
                    # print(ark)
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds[()], file_path)
                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    # print(ds[()].shape)
                    # print(ark)
                    self.data_info.append({
                        'file_path': file_path,
                        'type': dname,
                        'shape': ds[()].shape,
                        'cache_idx': idx
                    })

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path, 'r') as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    # idx = self._add_to_cache(ds.value, file_path)
                    idx = self._add_to_cache(ds[()], file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i, v in enumerate(self.data_info)
                                    if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{
                'file_path': di['file_path'],
                'type': di['type'],
                'shape': di['shape'],
                'cache_idx': -1
            } if di['file_path'] == removal_keys[0] else di
                              for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


def getHDF5dataLoader(data_path, loader_params, data_info_dict):
    print("Loading HDF5 Data from file:")
    print(data_path)
    time_load = time.time()
    dataset = HDF5Dataset(
        data_path,
        recursive=True,
        load_data=False,
        data_cache_size=4,
        data_info_dict=data_info_dict,
    )
    # dataset = HDF5Dataset(data_path, recursive=True, load_data=True, scaler=scaler)
    data_loader = data.DataLoader(dataset, **loader_params)
    if len(dataset) % loader_params["batch_size"] != 0:
        raise ValueError(
            "Size of dataset ({:}) has to be divisible with the batch size  ({:})."
            .format(len(dataset), loader_params["batch_size"]))
    time_load_end = time.time()
    printTime(time_load_end - time_load)
    return data_loader


def saveData(data, data_path, protocol):
    assert (protocol in ["hickle", "pickle"])
    if protocol == "hickle":
        saveDataHickle(data, data_path)
    elif protocol == "pickle":
        saveDataPickle(data, data_path)
    else:
        raise ValueError("Invalid protocol.")
    return 0


def loadData(data_path, protocol):
    assert (protocol in ["hickle", "pickle"])
    if protocol == "hickle":
        return loadDataHickle(data_path)
    elif protocol == "pickle":
        return loadDataPickle(data_path)
    else:
        raise ValueError("Invalid protocol.")


def saveDataHickle(data, data_path):
    data_path += ".hkl"
    hickle.dump(data, data_path)
    return 0


def saveDataPickle(data, data_path):
    data_path += ".pickle"
    with open(data_path, "wb") as file:
        # Pickle the "data" dictionary using the highest protocol available.
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        del data
    return 0


def loadDataPickle(data_path):
    data_path += ".pickle"
    print("Loading Data...")
    time_load = time.time()
    try:
        with open(data_path, "rb") as file:
            data = pickle.load(file)
    except Exception as inst:
        try:
            import pickle5 as pickle5
            with open(data_path, "rb") as file:
                data = pickle5.load(file)
        except Exception as inst:
            print("Datafile\n {:s}\nNOT FOUND.".format(data_path))
            raise ValueError(inst)
    time_load_end = time.time()
    printTime(time_load_end - time_load)
    return data


def loadDataHickle(data_path):
    data_path += ".hkl"
    print("Loading Data...")
    time_load = time.time()
    try:
        data = hickle.load(data_path)
    except Exception as inst:
        print("Datafile\n {:s}\nNOT FOUND.".format(data_path))
        raise ValueError(inst)
    time_load_end = time.time()
    printTime(time_load_end - time_load)
    return data


def printTime(seconds):
    time_str = utils_time.secondsToTimeStr(seconds)
    print("Time passed: {:}".format(time_str))
    return 0

'''
datasets.py: Customized dataset provider.
---- Dataset: object containing training set and test set
'''
import os
import pandas as pd
import numpy as np
import scipy.io as scio
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


def extract_d1(datapath):
    '''
    input: data path -> list
    output: X_train, y_train, X_test, y_test -> array-like object
    '''
    train_path = datapath[0]
    test_path = datapath[1]
    X_train = np.loadtxt(train_path[0])
    y_train = np.loadtxt(train_path[1])
    X_test = np.loadtxt(test_path[0])
    y_test = np.loadtxt(test_path[1])
    return X_train, y_train, X_test, y_test
    
    
class Dataset:
    '''
    Dataset of training set and test set
    '''
    
    def __init__(self, descr, datapath=None, extraction=extract_d1, preprocessing=False, persistency=False, artifact_dir="../artifacts/", LOAD=None):
        '''
        Parameters:
        -- descr: Description of dataset, e.g. the usage or components of dataset, which would be used in persistency directory -> str
        -- datapath: Path of the original data file where you want to load the data -> str
        -- extraction: Function need-to-implement, which is used to extract the array like data from origin data file -> function
        -- preprocessing: Use standard scaler to scale X and normalize y to 0 mean.
        -- persistency: Whether to execute persistency of the data loaded -> bool
        -- artifact_dir: Persistency top directory, e.g /home/xx/proj/data -> str
        -- LOAD: Load from existed pickle? cover the "extraction" and "datapath"-> bool
        '''
        
        self.descr = str(descr)
        if LOAD:
            if not artifact_dir.endswith("/"):
                artifact_dir = f"{artifact_dir}/"
            self.X_train = load(os.path.join(f"{artifact_dir}data/", LOAD, "X_train.pickle"))
            self.y_train = load(os.path.join(f"{artifact_dir}data/", LOAD, "y_train.pickle"))
            self.X_test = load(os.path.join(f"{artifact_dir}data/", LOAD, "X_test.pickle"))
            self.y_test = load(os.path.join(f"{artifact_dir}data/", LOAD, "y_test.pickle"))

        else:
            X_scaler = None
            y_mean = 0
            self.X_train, self.y_train, self.X_test, self.y_test = extraction(datapath)
            if preprocessing:
                # self.y_train = self.y_train.reshape(-1, 1)
                # self.y_test = self.y_test.reshape(-1, 1)
                self.X_scaler = StandardScaler()
                # self.y_scaler = StandardScaler()
                self.X_scaler.fit(self.X_train)
                # self.y_scaler.fit(self.y_train)
                self.X_train = self.X_scaler.transform(self.X_train)
                # self.y_train = self.y_scaler.transform(self.y_train)
                self.X_test = self.X_scaler.transform(self.X_test)
                # self.y_test = self.y_scaler.transform(self.y_test)
                # self.y_train = self.y_train.reshape(-1,)
                # self.y_test = self.y_test.reshape(-1,)
            if persistency:
                PREFIX = "data/{}/".format(self.descr)
                if not os.path.exists(os.path.join(artifact_dir, PREFIX)):
                    os.mkdir(os.path.join(artifact_dir, PREFIX))
                if preprocessing and X_scaler:
                    dump(X_scaler, os.path.join(artifact_dir, PREFIX, "Xscaler_ymean_{}.pickle".format(y_mean)))
                dump(self.X_train, os.path.join(artifact_dir, PREFIX, "X_train.pickle"))
                dump(self.y_train, os.path.join(artifact_dir, PREFIX, "y_train.pickle"))
                dump(self.X_test, os.path.join(artifact_dir, PREFIX, "X_test.pickle"))
                dump(self.y_test, os.path.join(artifact_dir, PREFIX, "y_test.pickle"))
        
    def convert(self, to_type):
        '''
        Convert self.xxx to relavant type:
        Parameters:
        -- to_type: The type you want to transform, only support [np.ndarray, torch.Tensor, pd.DataFrame] -> type
        '''
        _type_str = ["Array", "Tensor", "DataFrame"]
        _type = [(np.ndarray, np.generic), torch.Tensor, pd.DataFrame]
        if to_type not in _type_str:
            raise ValueError("to_type must be one of {}".format(_type_str))
        transform = [lambda a: t(a) for t in [np.array, torch.Tensor, pd.DataFrame]]
        for i, _to_type in enumerate(_type_str):
            if to_type == _to_type:
                if not isinstance(self.X_train, _type[i]):
                    try:
                        self.X_train = transform[i](self.X_train)
                    except:
                        raise ValueError("cannot transform to {}".format(_type_str[i]))
                if not isinstance(self.y_train, _type[i]):
                    try:
                        self.y_train = transform[i](self.y_train)
                    except:
                        raise ValueError("cannot transform to {}".format(_type_str[i]))
                if not isinstance(self.X_test, _type[i]):
                    try:
                        self.X_test = transform[i](self.X_test)
                    except:
                        raise ValueError("cannot transform to {}".format(_type_str[i]))
                if not isinstance(self.y_test, _type[i]):
                    try:
                        self.y_test = transform[i](self.y_test)
                    except:
                        raise ValueError("cannot transform to {}".format(_type_str[i]))
                        
    def reverse_scaler(self, y):
        return self.y_scaler.inverse_transform(y)
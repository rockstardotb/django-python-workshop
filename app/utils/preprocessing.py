# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
def import_data(csv, xidx, yidx, start=0, header=0):
    dataset = ""
    if header is None:
        dataset = pd.read_csv(csv,header=header)
    else:
        dataset = pd.read_csv(csv)
    X = dataset.iloc[:, start:xidx].values
    y = dataset.iloc[:, yidx].values
    return X,y

# Taking care of missing data
def fix_missing(X, xstart, xstop):
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, xstart:xstop])
    X[:, xstart:xstop] = imputer.transform(X[:, xstart:xstop])
    return X

# Encoding categorical data
def categorical_encode(data, independent=True, equal=True, idx=0):
    # Encoding the Independent Variable
    # [string1,string2,string3] --> [0,1,2]
    if independent:
        labelencoder_data = LabelEncoder()
        data[:, idx] = labelencoder_data.fit_transform(data[:, idx])

        if equal:
            # Prevent machine from thinking one category is greater than
            # another
            # [0,1,2] --> [[1,0,0],[0,1,0],[0,0,1]]
            # first column --> France, second column --> Germany, third column --> Spain
            onehotencoder = OneHotEncoder(categorical_features = [idx])
            data = onehotencoder.fit_transform(data).toarray()

    else:
        # Encoding the Dependent Variable
        # Dependent variable doesn't need OneHotEncoder
        # ['No','Yes'] --> [0,1]
        labelencoder_data = LabelEncoder()
        data = labelencoder_data.fit_transform(data)

    return data

# Split dataset into training and test sets
def create_sets(X, y, size=0.2, random_state=0):

        return train_test_split(X, y, test_size = size, random_state = random_state)

# Feature scaling
def feature_scale(X_train, X_test=None):
        # Put columns in same scale so one feature doesn't
        # dominate another

        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        if X_test is None:
            return X_train, sc_X

        X_test = sc_X.transform(X_test)

        return X_train, X_test, sc_X


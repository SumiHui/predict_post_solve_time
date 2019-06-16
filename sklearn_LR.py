from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors.regression import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.externals import joblib

import h5py
import numpy as np
import random

def get_xy(filepath="dataset/train.hdf5"):

    dataset = h5py.File(filepath, "r")
    body_ndarray = dataset['body'].value
    tags_ndarray = dataset['tags'].value
    rate_ndarray = dataset['rate'].value
    time_ndarray = dataset['time'].value
    title_ndarray = dataset['title'].value
    week_ndarray = dataset['week'].value

    # delete over 4 days
    body_ndarray = body_ndarray[time_ndarray <= 4e5]
    tags_ndarray = tags_ndarray[time_ndarray <= 4e5]
    rate_ndarray = rate_ndarray[time_ndarray <= 4e5]
    title_ndarray = title_ndarray[time_ndarray <= 4e5]
    week_ndarray = week_ndarray[time_ndarray <= 4e5]
    time_ndarray = time_ndarray[time_ndarray <= 4e5]  # delete samples that time over 4e5, attention: must put on this

    # shuffle data, `fancy indexing`, the info/label data is correctly connected to each set of features
    indexes = np.arange(time_ndarray.shape[0])
    random.shuffle(indexes)
    body_ndarray = np.take(body_ndarray, indexes, axis=0)
    tags_ndarray = np.take(tags_ndarray, indexes, axis=0)
    rate_ndarray = np.take(rate_ndarray, indexes, axis=0)
    time_ndarray = np.take(time_ndarray, indexes, axis=0)
    week_ndarray = np.take(week_ndarray, indexes, axis=0)
    title_ndarray = np.take(title_ndarray, indexes, axis=0)

    time_max = np.asarray([4e5])

    time_ndarray = time_ndarray / time_max  # (time_ndarray - 0) / (time_max - 0)
    print("time var:{}".format(np.var(time_ndarray)))

    rate_ndarray=np.expand_dims(rate_ndarray,-1)
    week_ndarray=np.expand_dims(week_ndarray,-1)
    X=np.concatenate([body_ndarray,title_ndarray,tags_ndarray,rate_ndarray,week_ndarray],-1)
    y=time_ndarray
    return X,y



def lr(X,y):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "sklearn_model_params/LinearRegression.m")

# RandomForestRegressor
def reg(X,y):
    model = RandomForestRegressor()
    model.fit(X,y)
    joblib.dump(model, "sklearn_model_params/RandomForestRegressor.m")

# KNeighborsRegressor
def knr(X,y):
    model = KNeighborsRegressor()
    model.fit(X, y)
    joblib.dump(model, "sklearn_model_params/KNeighborsRegressor.m")

#  SVM
def svr(X,y):
    model = SVR()
    model.fit(X, y)
    joblib.dump(model, "sklearn_model_params/SVR.m")

def mlp(X,y):
    model = MLPRegressor()
    model.fit(X, y)
    joblib.dump(model, "sklearn_model_params/MLPRegressor.m")

def test(X,y,params_file="sklearn_model_params/LinearRegression.m"):
    model=joblib.load(params_file)
    y_hat=model.predict(X)
    mse=np.mean(np.square(y-y_hat))
    print(mse)


if __name__ == '__main__':
    # X_train, y_train = get_xy()
    # print(X_train.shape, y_train.shape)
    # # lr(X_train,y_train)
    # # reg(X_train,y_train)
    # # print("finish reg")
    # #
    # # knr(X_train,y_train)
    # # print("finish knr")
    # #
    # # svr(X_train,y_train)
    # # print("finish svr")
    # mlp(X_train,y_train)
    # del X_train,y_train

    X_test, y_test = get_xy("dataset/test.hdf5")
    print("test: {}/{}".format(X_test.shape, y_test.shape))
    # test(X_test, y_test, "sklearn_model_params/RandomForestRegressor.m")
    # test(X_test, y_test, "sklearn_model_params/KNeighborsRegressor.m")
    # test(X_test, y_test, "sklearn_model_params/SVR.m")
    test(X_test, y_test, "sklearn_model_params/MLPRegressor.m")
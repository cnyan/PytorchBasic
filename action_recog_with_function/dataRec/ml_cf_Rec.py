# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2021/1/20 23:13
@Describe：

"""


import joblib
import AUtils
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier


class ML_Features_Rec():
    def __init__(self, model, model_name, axis):
        self.data = self.read_features(axis)
        self.model = model
        self.model_name = model_name
        self.axis = axis
        print(f'=========当前模式 {model_name}-{axis}=============')

    def read_features(self,axis):
        train_features_path = fr'src/ml_cf_features/train_features_mat-{axis}.npy'
        test_features_path = fr'src/ml_cf_features/test_features_mat-{axis}.npy'

        train_data = np.load(train_features_path)  # [7000 rows x 181 columns]
        test_data = np.load(test_features_path)  # [1000 rows x 181 columns]

        print(f'train data shape:{train_data.shape}')
        print(f'test data shape:{test_data.shape}')
        return train_data, test_data

    def train(self):
        train_data, test_data = self.data
        Xtrain = train_data[:, :-1]
        ytrain = train_data[:, -1]

        train_model = self.model.fit(Xtrain, ytrain)
        joblib.dump(train_model, fr'src/ml_cf_model/{self.model_name}_model-{self.axis}.pkl')

    def predict(self):
        train_data, test_data = self.data
        Xtest = test_data[:, :-1]
        ytest = test_data[:, -1]

        knn_model = joblib.load(fr'src/ml_cf_model/{self.model_name}_model-{self.axis}.pkl')
        y_predict = knn_model.predict(Xtest)

        AUtils.plot_confusion_matrix(ytest, y_predict, ['Action0', 'Action1', 'Action2', 'Action3', 'Action4'],
                                     fr'src/ml_cf_plt_img/{self.model_name}_predict-{self.axis}.jpg',
                                     title=fr'{self.model_name}-{self.axis} Confusion matrix')
        AUtils.metrics(ytest, y_predict)


if __name__ == '__main__':

    for axis in ['9axis', '6axis']:
        knn_model = ML_Features_Rec(KNeighborsClassifier(n_neighbors=5), 'KNeighbors', axis=axis)
        # knn_model.train()
        knn_model.predict()

        svm_model = ML_Features_Rec(SVC(kernel='rbf', class_weight='balanced'), 'SVM', axis=axis)
        # svm_model.train()
        svm_model.predict()

        nb_model = ML_Features_Rec(MultinomialNB(), 'MultinomialNB', axis=axis)
        nb_model.train()
        nb_model.predict()

        rf_model = ML_Features_Rec(RandomForestClassifier(), 'RandomForest', axis=axis)
        # rf_model.train()
        rf_model.predict()

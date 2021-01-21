# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2021/1/21 15:35
@Describe：

"""


import joblib
import AUtils
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')


class ML_Features_Rec():
    def __init__(self, model, model_name, axis):
        self.data = self.read_features(axis)
        self.model = model
        self.model_name = model_name
        self.axis = axis
        print(f'=========当前模式 {model_name}-{axis}=============')

    def read_features(self,axis):
        test_features_path = fr'src/ml_cf_features/other_test_features_mat-{axis}.npy'
        test_data = np.load(test_features_path)  # [1000 rows x 181 columns]

        print(f'test data shape:{test_data.shape}')
        return test_data

    def predict(self):
        test_data = self.data
        Xtest = test_data[:, :-1]
        ytest = test_data[:, -1]

        knn_model = joblib.load(fr'src/ml_cf_model/{self.model_name}_model-{self.axis}.pkl')
        y_predict = knn_model.predict(Xtest)

        AUtils.plot_confusion_matrix(ytest, y_predict, [0, 1, 2, 3, 4],
                                     fr'src/ml_cf_plt_img/{self.model_name}_others_test_predict-{self.axis}.jpg',
                                     title=fr'{self.model_name}-{self.axis} Confusion matrix')
        AUtils.metrics(ytest, y_predict)


if __name__ == '__main__':

    for axis in ['9axis', '6axis']:
        knn_model = ML_Features_Rec(KNeighborsClassifier(n_neighbors=1), 'knn', axis=axis)
        knn_model.predict()

        svm_model = ML_Features_Rec(SVC(kernel='rbf', class_weight='balanced'), 'svc', axis=axis)
        svm_model.predict()

        nb_model = ML_Features_Rec(GaussianNB(), 'nb', axis=axis)
        nb_model.predict()

        rf_model = ML_Features_Rec(RandomForestClassifier(), 'rf', axis=axis)
        rf_model.predict()

# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2021/1/21 16:30
@Describe：
测试集是实验室其他同学的动作数据
"""

import joblib
import AUtils
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


class ML_Features_Rec():
    def __init__(self, model, model_name, axis):

        self.model = model
        self.model_name = model_name
        self.axis = axis
        self.data = self.read_features(axis)

        print(f'=========当前模式 {model_name}-{axis}=============')

    def read_features(self,axis):
        test_features_path = fr'src/ml_tf_Features/other_test_features_mat-{axis}.npy'

        test_data = np.load(test_features_path)  # [1000 rows x 181 columns]

        print(f'test data shape:{test_data.shape}')
        return test_data


    def predict(self):
        test_data = self.data
        Xtest = test_data[:, :-1]
        ytest = test_data[:, -1]

        knn_model = joblib.load(fr'src/ml_tf_model/{self.model_name}_model-{self.axis}.pkl')
        y_predict = knn_model.predict(Xtest)

        AUtils.plot_confusion_matrix(ytest, y_predict, ['Action0', 'Action1', 'Action2', 'Action3', 'Action4'],
                                     fr'src/ml_tf_plt_img/{self.model_name}_amateur_predict-{self.axis}.jpg',
                                     title=fr'{self.model_name}-{self.axis} Confusion matrix')
        AUtils.metrics(ytest, y_predict)
        return ytest, y_predict


if __name__ == '__main__':

    for axis in ['9axis', '6axis']:
        true_predict_dict = {}

        knn_model = ML_Features_Rec(KNeighborsClassifier(n_neighbors=5), 'KNN', axis=axis)
        y_label,y_predict = knn_model.predict()
        true_predict_dict['Knn_Confusion_Matrix'] = (y_label, y_predict)

        svm_model = ML_Features_Rec(SVC(kernel='rbf', class_weight='balanced'), 'SVM', axis=axis)
        y_label,y_predict = svm_model.predict()
        true_predict_dict['SVM_Confusion_Matrix'] = (y_label, y_predict)

        nb_model = ML_Features_Rec(GaussianNB(), 'GaussianNB', axis=axis)
        y_label,y_predict = nb_model.predict()
        true_predict_dict['GaussianNB_Confusion_Matrix'] = (y_label, y_predict)

        rf_model = ML_Features_Rec(RandomForestClassifier(), 'RandomForest', axis=axis)
        y_label,y_predict = rf_model.predict()
        true_predict_dict['RandomForest_Confusion_Matrix'] = (y_label, y_predict)

        # 开始画图
        index = 1
        plt.figure(figsize=(10, 10))

        classes = ['Action0', 'Action1', 'Action2', 'Action3', 'Action4']
        savePath = f'src/ml_tf_plt_img/Classifier_Confusion_Matrix_amateur_{axis}.png'
        for key, (y_label, y_predict) in true_predict_dict.items():

            plt.subplot(220 + index)
            index += 1

            cm = confusion_matrix(y_label, y_predict)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(key)
            # plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)

            plt.ylim(len(cm) - 0.5, -0.5)
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.15)
            plt.xlabel('True label')
            plt.ylabel('Predicted label')

        plt.savefig(savePath,bbox_inches='tight',dpi=100)
        plt.show()
        plt.close()
# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2021/1/20 20:25
@Describe：

"""

import joblib
import AUtils
import numpy as np
import pandas as pd
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class ML_Features_Rec():
    def __init__(self, model, model_name, axis):
        self.data = self.read_features(axis)
        self.model = model
        self.model_name = model_name
        self.axis = axis
        print(f'=========当前模式 {model_name}-{axis}=============')

    def read_features(self,axis):
        train_features_path = fr'src/ml_tf_Features/train_features_mat-{axis}.npy'
        test_features_path = fr'src/ml_tf_Features/test_features_mat-{axis}.npy'

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
        joblib.dump(train_model, fr'src/ml_tf_model/{self.model_name}_model-{self.axis}.pkl')

    def predict(self):
        train_data, test_data = self.data
        Xtest = test_data[:, :-1]
        ytest = test_data[:, -1]

        knn_model = joblib.load(fr'src/ml_tf_model/{self.model_name}_model-{self.axis}.pkl')
        y_predict = knn_model.predict(Xtest)

        AUtils.plot_confusion_matrix(ytest, y_predict, ['Action0', 'Action1', 'Action2', 'Action3', 'Action4'],
                                     fr'src/ml_tf_plt_img/{self.model_name}_predict-{self.axis}.jpg',
                                     title=fr'{self.model_name} Classifier Confusion Matrix')
        AUtils.metrics(ytest, y_predict)
        return ytest,y_predict


if __name__ == '__main__':

    for axis in ['6axis', '9axis']:
        true_predict_dict = {}

        knn_model = ML_Features_Rec(KNeighborsClassifier(n_neighbors=5), 'KNN', axis=axis)
        # knn_model.train()
        y_label,y_predict = knn_model.predict()
        true_predict_dict['Knn_Confusion_Matrix'] = (y_label,y_predict)

        svm_model = ML_Features_Rec(SVC(kernel='rbf', class_weight='balanced'), 'SVM', axis=axis)
        # svm_model.train()
        y_label,y_predict = svm_model.predict()
        true_predict_dict['SVM_Confusion_Matrix'] = (y_label, y_predict)

        nb_model = ML_Features_Rec(GaussianNB(), 'GaussianNB', axis=axis)
        # nb_model.train()
        y_label,y_predict = nb_model.predict()
        true_predict_dict['GaussianNB_Confusion_Matrix'] = (y_label, y_predict)

        rf_model = ML_Features_Rec(RandomForestClassifier(), 'RandomForest', axis=axis)
        # rf_model.train()
        y_label,y_predict = rf_model.predict()
        true_predict_dict['RandomForest_Confusion_Matrix'] = (y_label, y_predict)

        # 开始画图
        index = 1
        plt.figure(figsize=(10,10), dpi=200)

        classes = ['Action0', 'Action1', 'Action2', 'Action3', 'Action4']
        savePath = f'src/ml_tf_plt_img/Classifier_Confusion_Matrix_Major_{axis}.png'
        for key,(y_label, y_predict) in true_predict_dict.items():

            plt.subplot(220+index)
            index +=1

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
        plt.savefig(savePath,bbox_inches='tight')
        plt.show()
        plt.close()


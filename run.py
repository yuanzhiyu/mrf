# encoding: utf-8

import warnings
import sys
warnings.filterwarnings('ignore')
from time import time
from MultinomialRF import MultinomialRF
import numpy as np
from sklearn.model_selection import KFold
from utils import load_data, output_time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_test(train_set, test_set, feature_attr):
    # Train and test respectively.

    # Create Multinomial Random Forest, which is similar to use scikit-learn's API.
    clf = MultinomialRF(n_estimators=100,
                        min_samples_leaf=5,
                        B1=10,
                        B2=10,
                        B3=None,
                        partition_rate=1,
                        n_jobs=10)

    start_time = time()
    train_accuracy = clf.fit(train_set[:, :-1], train_set[:, -1], feature_attr)
    end_time = time()
    training_time = end_time - start_time

    pred = clf.predict(test_set[:, :-1])
    test_accuracy = accuracy_score(pred, test_set[:, -1].astype(int))
    print("训练耗时: {:.2f}s, 训练精度: {:.4f}, 测试精度: {:.4f}".format(training_time, train_accuracy, test_accuracy))

    return train_accuracy, test_accuracy


def cross_validation(data, feature_attr, ROUND_NUM=10, FOLD_NUM=10):
    # Cross validation with the input data.
    print("{0}-fold cross validation with {1} times.".format(str(FOLD_NUM), str(ROUND_NUM)))

    train_res = []
    test_res = []
    for i in range(ROUND_NUM):
        kf = KFold(n_splits=FOLD_NUM, shuffle=True)
        for train_index, test_index in kf.split(X=data[:, :-1], y=data[:, -1], groups=data[:, -1]):
            train_set, test_set = data[train_index], data[test_index]
            train_acc, test_acc = train_test(train_set, test_set, feature_attr)
            train_res.append(train_acc)
            test_res.append(test_acc)

        print("ROUND[{0}] 平均训练精度: {1}".format(str(i + 1), str(np.mean(train_res[-FOLD_NUM:]))[:6]))
        print("ROUND[{0}] 平均测试精度: {1}".format(str(i + 1), str(np.mean(test_res[-FOLD_NUM:]))[:6]))
        print("ROUND[{0}] 测试精度标准差: {1}".format(str(i + 1), str(np.std(test_res[-FOLD_NUM:]))[:6]))

    return np.mean(train_res), np.mean(test_res), np.std(test_res)


if __name__ == "__main__":

    if len(sys.argv) < 6:
        print("Usage: python run.py [dataset] [frac_num] [cross_validation] [round_num] [fold_num]")
        exit(0)

    dataset = sys.argv[1]
    frac_num = float(sys.argv[2])
    data, feature_attr = load_data(dataset, frac_num)
    np.random.shuffle(data)
    print("Data Size:", data.shape)

    CROSS_VALIDATION = int(sys.argv[3])   # run as cross validation or not

    if CROSS_VALIDATION:
        round_num = int(sys.argv[4])
        fold_num = int(sys.argv[5])
        start_time = time()
        train_res, test_res, test_std = cross_validation(data, feature_attr, round_num, fold_num)
        end_time = time()
        evaluate_time = end_time - start_time
        print("总耗时: {:.2f}s, 平均训练精度: {:.4f}, 平均测试精度: {:.4f}, 测试精度标准差: {:.4f}".format(evaluate_time, train_res, test_res, test_std))
    else:
        train_set, test_set = train_test_split(data, test_size=0.1, shuffle=True)
        train_res, test_res = train_test(train_set, test_set, feature_attr)

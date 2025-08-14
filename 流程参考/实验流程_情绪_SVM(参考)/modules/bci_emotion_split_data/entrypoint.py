import os

import numpy as np
import sklearn.svm as svm


def main(all_data: np.ndarray, all_label: np.ndarray):
    print("all_data 类型:", type(all_data))
    print("all_data 形状:", all_data.shape)
    print("all_label 类型:", type(all_label))
    print("all_label 形状:", all_label.shape)

    num_subjects = all_data.shape[0]
    split_index = int(num_subjects * 0.8)

    train_subjects = list(range(split_index))
    test_subjects = list(range(split_index, num_subjects))
    print("train_subjects: ", train_subjects)
    print("test_subjects: ", test_subjects)

    train_data = all_data[train_subjects]
    test_data = all_data[test_subjects]

    train_label = all_label[train_subjects]
    test_label = all_label[test_subjects]

    # 将训练集和测试集数据扁平化为 (num_samples, num_channels*window_size)
    X_train = train_data.reshape(-1, train_data.shape[3] * train_data.shape[4])
    X_test = test_data.reshape(-1, test_data.shape[3] * test_data.shape[4])

    # 将训练集和测试集标签扁平化为 (num_samples)
    y_train = train_label.reshape(-1)
    y_test = test_label.reshape(-1)

    print("X_train 类型:", type(X_train))
    print("X_train 形状:", X_train.shape)
    print("y_train 类型:", type(y_train))
    print("y_train 形状:", y_train.shape)

    return [X_train, y_train, X_test, y_test]


if __name__ == "__main__":
    import pickle

    os.environ["INPUT_PARAM_C"] = "1"
    os.environ["INPUT_PARAM_DECISION_FUNCTION_SHAPE"] = "ovo"

    neural_data = pickle.load(open("../load_sample_data_0.pkl", "rb"))
    target = pickle.load(open("../load_sample_data_1.pkl", "rb"))

    [model] = main(neural_data, target)

    pickle.dump(model, open("../svm_fit_model_0.pkl", "wb"))

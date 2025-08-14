import os

import numpy as np
import sklearn.svm as svm


def main(x_train: np.ndarray, y_train: np.ndarray):
    kernel = os.environ.get("PARAM_SVC_KERNEL")
    model = svm.SVC(kernel=kernel)  # 选择线性SVM
    print("Start training SVM model...")

    print("X_train 类型:", type(x_train))
    print("X_train 形状:", x_train.shape)
    print("y_train 类型:", type(y_train))
    print("y_train 形状:", y_train.shape)

    model.fit(x_train, y_train)

    return [model]


if __name__ == "__main__":
    import pickle

    os.environ["INPUT_PARAM_C"] = "1"
    os.environ["INPUT_PARAM_DECISION_FUNCTION_SHAPE"] = "ovo"

    neural_data = pickle.load(open("../load_sample_data_0.pkl", "rb"))
    target = pickle.load(open("../load_sample_data_1.pkl", "rb"))

    [model] = main(neural_data, target)

    pickle.dump(model, open("../svm_fit_model_0.pkl", "wb"))

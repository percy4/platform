import os

import numpy as np
import sklearn.svm as svm
from sklearn.metrics import accuracy_score, f1_score


def main(y_test: np.ndarray, y_pred: np.ndarray):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")
    return [accuracy]


if __name__ == "__main__":
    import pickle

    os.environ["INPUT_PARAM_C"] = "1"
    os.environ["INPUT_PARAM_DECISION_FUNCTION_SHAPE"] = "ovo"

    neural_data = pickle.load(open("../load_sample_data_0.pkl", "rb"))
    target = pickle.load(open("../load_sample_data_1.pkl", "rb"))

    [model] = main(neural_data, target)

    pickle.dump(model, open("../MSCT_fit_model_0.pkl", "wb"))

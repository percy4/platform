import os

import numpy as np
import sklearn.svm as svm


def main(data: np.ndarray):
    num_subs=data.shape[0]
    num_windows=data.shape[2]
    
    num_negative = 4
    num_positive = 4
    y_label = []
    for i in range(0, num_negative):
        y_label.extend([i] * 3)
    y_label.extend([num_negative] * 4)
    for i in range(num_negative + 1, num_negative + num_positive + 1):
        y_label.extend([i] * 3)
    y_label = np.array(y_label)
    y_label = np.repeat(y_label[:, np.newaxis], num_windows, axis=1)

    # 扩展为 (num_subs, num_trials, num_windows)
    y_label = np.tile(y_label, (num_subs, 1, 1))

    return [y_label]


if __name__ == "__main__":
    import pickle

    os.environ["INPUT_PARAM_C"] = "1"
    os.environ["INPUT_PARAM_DECISION_FUNCTION_SHAPE"] = "ovo"

    neural_data = pickle.load(open("../load_sample_data_0.pkl", "rb"))
    target = pickle.load(open("../load_sample_data_1.pkl", "rb"))

    [model] = main(neural_data, target)

    pickle.dump(model, open("../svm_fit_model_0.pkl", "wb"))

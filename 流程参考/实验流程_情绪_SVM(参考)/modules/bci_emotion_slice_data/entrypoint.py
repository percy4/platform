import os

import numpy as np
import sklearn.svm as svm


def main(data: np.ndarray):
    window_size = int(os.environ.get("PARAM_WINDOW"))
    print("window_size: ", window_size)
    step = int(os.environ.get("PARAM_STEP"))
    print("step: ", step)
    num_subs, num_trials, num_channels, num_timepoints = data.shape
    num_windows = (num_timepoints - window_size) // step + 1
    windows = []
    for k in range(num_windows):
        start_idx = k * step
        end_idx = start_idx + window_size
        window_data = data[:, :, :, start_idx:end_idx]  # 选取窗口数据
        windows.append(window_data)
    # 返回的shape为(num_subs, num_trials, num_windows, num_channels, window_size)
    windows = np.array(windows).transpose(1, 2, 0, 3, 4)
    # print("windows: ", windows)
    print("windows shape: ", windows.shape)
    return [windows]


if __name__ == "__main__":
    pass
    # import pickle

    # os.environ["INPUT_PARAM_C"] = "1"
    # os.environ["INPUT_PARAM_DECISION_FUNCTION_SHAPE"] = "ovo"

    # neural_data = pickle.load(open("../load_sample_data_0.pkl", "rb"))
    # target = pickle.load(open("../load_sample_data_1.pkl", "rb"))

    # [model] = main(neural_data, target)

    # pickle.dump(model, open("../svm_fit_model_0.pkl", "wb"))

import os

import numpy as np
import sklearn.svm as svm
import pickle


def main():
    data_path = os.environ.get("PARAM_LOAD_PATH")
    num_subs = int(os.environ.get("PARAM_NUM_SUBS"))
    num_trials = int(os.environ.get("PARAM_NUM_TRIALS"))
    num_channels = int(os.environ.get("PARAM_NUM_CHANNELS"))
    num_timepoints = int(os.environ.get("PARAM_NUM_TIMEPOINTS"))

    print("num_subs: ", num_subs)
    print("num_trials: ", num_trials)
    print("num_channels: ", num_channels)
    print("num_timepoints: ", num_timepoints)

    file_name = os.listdir(data_path)
    file_name.sort()

    all_data = np.zeros((num_subs, num_trials, num_channels, num_timepoints))
    for i in range(num_subs):
        data = pickle.load(open(os.path.join(data_path, file_name[i]), "rb"))
        all_data[i] = data

    return [all_data]


if __name__ == "__main__":
    # import pickle

    # os.environ["INPUT_PARAM_C"] = "1"
    # os.environ["INPUT_PARAM_DECISION_FUNCTION_SHAPE"] = "ovo"

    # neural_data = pickle.load(open("../load_sample_data_0.pkl", "rb"))
    # target = pickle.load(open("../load_sample_data_1.pkl", "rb"))

    # [model] = main(neural_data, target)

    # pickle.dump(model, open("../svm_fit_model_0.pkl", "wb"))
    pass

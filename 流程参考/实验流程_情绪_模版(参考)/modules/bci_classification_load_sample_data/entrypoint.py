import os

import scipy
import re
import numpy as np


def main():
    mat_path = os.environ.get("INPUT_MAT_PATH")
    label_txt_path = os.environ.get("INPUT_LABEL_TXT_PATH")

    print(f"mat_path: {mat_path}")
    print(f"label_txt_path: {label_txt_path}")

    neural_data = scipy.io.loadmat(mat_path)["NeuralData"]
    with open(
        label_txt_path,
        "r",
    ) as f:
        pattern = (
            r"NO.(\d+)\s+START BIN: (\d+)\s+END BIN: (\d+)\s+DIRECTION: ([\d|\*]+)\s+"
        )
        label_data = re.findall(pattern, f.read(), re.M)
    _index, start_bin, end_bin, direction = zip(
        *[(0 if v == "*" else int(v) for v in trial) for trial in label_data]
    )
    neural_data = [neural_data[:, s - 1 : e].T for s, e in zip(start_bin, end_bin)]
    return [np.array(neural_data), np.array(direction)]


if __name__ == "__main__":
    import pickle

    home = os.environ.get("HOME")
    mat_path = f"{home}/Downloads/sample-module-svm-lr-mlp-rnn/Dataset-small/training/NeuralData1.mat"
    os.environ["INPUT_MAT_PATH"] = mat_path
    label_path = f"{home}/Downloads/sample-module-svm-lr-mlp-rnn/Dataset-small/training/label1.txt"
    os.environ["INPUT_LABEL_TXT_PATH"] = label_path

    [data, direction] = main()

    with open("../load_sample_data_0.pkl", "wb") as pkl:
        pickle.dump(data, pkl)
    with open("../load_sample_data_1.pkl", "wb") as pkl:
        pickle.dump(direction, pkl)

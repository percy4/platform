import os

import numpy


def main(input_0: numpy.ndarray, input_1: int):
    # 参数
    env_str = os.environ.get("INPUT_STRING")
    env_int = int(os.environ.get("INPUT_INT"))
    env_float = float(os.environ.get("INPUT_FLOAT"))
    env_bool = bool(os.environ.get("INPUT_BOOL"))

    output_0 = None
    output_1 = None
    # code

    return [output_0, output_1]


if __name__ == "__main__":
    import pickle

    os.environ["INPUT_STRING"] = "ABC"
    os.environ["INPUT_INT"] = "1"
    os.environ["INPUT_FLOAT"] = "1.23"
    os.environ["INPUT_BOOL"] = "True"

    input_data_0 = pickle.load(open("../input_data_0.pkl", "rb"))
    input_data_1 = pickle.load(open("../input_data_1.pkl", "rb"))

    [output_data_0, output_data_1] = main(input_data_0, input_data_1)

    pickle.dump(output_data_0, open("../output_data_0.pkl", "wb"))
    pickle.dump(output_data_1, open("../output_data_1.pkl", "wb"))

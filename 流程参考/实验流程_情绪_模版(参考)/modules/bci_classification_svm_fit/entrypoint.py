import os

import numpy
import sklearn.svm as svm


def main(neural_data: numpy.ndarray, target: numpy.ndarray):
    # Get parameters from environment
    env_c = int(os.environ.get("INPUT_PARAM_C"))
    env_dfs = os.environ.get("INPUT_PARAM_DECISION_FUNCTION_SHAPE")

    print(f"C: {env_c}")
    print(f"decision_function_shape: {env_dfs}")

    model = svm.SVC(C=env_c, decision_function_shape=env_dfs)
    model.fit(neural_data, target)

    print(f"model type: {type(model)}")

    return [model]


if __name__ == "__main__":
    import pickle

    os.environ["INPUT_PARAM_C"] = "1"
    os.environ["INPUT_PARAM_DECISION_FUNCTION_SHAPE"] = "ovo"

    neural_data = pickle.load(open("../load_sample_data_0.pkl", "rb"))
    target = pickle.load(open("../load_sample_data_1.pkl", "rb"))

    [model] = main(neural_data, target)

    pickle.dump(model, open("../svm_fit_model_0.pkl", "wb"))

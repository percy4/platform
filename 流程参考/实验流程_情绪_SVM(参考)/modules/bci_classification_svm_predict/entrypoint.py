import numpy
import sklearn.svm


def main(model: sklearn.svm, data: numpy.ndarray):
    print(f"model type: {type(model)}")
    print(f"data type: {type(data)}")

    result = model.predict(data)

    print(f"result shape: {result.shape}")
    print(f"result type: {type(result)}")
    return [result]


if __name__ == "__main__":
    import pickle

    model = pickle.load(open("../svm_fit_model_0.pkl", "rb"))
    data = pickle.load(open("../load_sample_data_0.pkl", "rb"))

    result = main(model, data)

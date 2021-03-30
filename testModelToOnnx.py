from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import pickle
import numpy as np
import DataBaseHandler
import ApiParser


def save_as_onnx(model_to_save, filename):
    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(model_to_save, initial_types=initial_type)
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())


def save_model_as_pickle(model_to_save, filename):
    pickle.dump(model_to_save, open(filename, "wb"))


def load_model_from_pickle(filename):
    model = pickle.load(open(filename, "rb"))
    return model


def get_samples_to_nparray(sys_samples_cursor):
    list1 = []
    list2 = []
    for sample in sys_samples_cursor:
        dia_sample = DataBaseHandler.get_corresponding_bp_sample(sample)
        user_id = sample['user']
        sys_sample_value = sample["value"]
        dia_sample_value = dia_sample["value"]
        user = DataBaseHandler.get_user(user_id)

        list1.append([sys_sample_value, dia_sample_value])
        list2.append(user["age"])
    x_array = np.array(list1)
    y_array = np.array(list2)
    return x_array, y_array


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)


def tune_model(model, x_tune, y_tune):
    model.patial_fit()


def send_api_request():
    print("Checking DT-API for data.")
    new_samples = ApiParser.get_data()
    if len(new_samples) > 0:
        print("New samples found.")
        sys_samples = []
        for s in new_samples:
            if s['type'] == "systolicBloodPressure":
                sys_samples.append(s)

        print(sys_samples)
        return sys_samples
    else:
        print("No new samples found.")


my_model = LinearRegression()

#systolics = DataBaseHandler.get_samples(type="systolicBloodPressure")
#X, Y = get_samples_to_nparray(systolics)

print("-- DATA FROM DB --")
#print(X)
#print(Y)

#train_model(my_model, X, Y)



print("Trained")
#print(my_model.coef_)
#print(my_model.predict([[130, 83]]))

model_filip = LinearRegression()

model_filip.intercept_ = -15.139611
model_filip.coef_ = np.array([[0.048337, 0.055844, 0.060932]])

print("filip model predict: ")
print(model_filip.predict([[24, 130, 83]]))

#save_as_onnx(my_model, "myModel.onnx")
#save_as_onnx(model_filip, "model.onnx")
save_model_as_pickle(model_filip, "model.pkl")

sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
print(input_name)
print(label_name)
test_x = np.array([[24, 130, 83, 1]])
prediction = sess.run(None, {input_name: test_x.astype(np.float32)})[0]

print(prediction)




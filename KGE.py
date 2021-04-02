import DataBaseHandler
import ModelHandler
import time
from sklearn.linear_model import LinearRegression


model = LinearRegression()

sys_samples = DataBaseHandler.get_samples(type="systolicBloodPressure")
x_train, y_train = ModelHandler.get_samples_to_nparray(sys_samples)
ModelHandler.train_model(model, x_train, y_train)

while True:
    new_systolic_samples = ModelHandler.get_new_systolic_samples_from_API()
    if new_systolic_samples is not None:
        x_tune, y_tune = ModelHandler.get_samples_to_nparray(new_systolic_samples)
        ModelHandler.tune_model(model, x_tune, y_tune)
        ModelHandler.save_as_onnx(model, "MyModel.onnx")
        ModelHandler.save_model_as_pickle(model, "MyModel.pkl")
    time.sleep(60)



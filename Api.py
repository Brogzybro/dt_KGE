from flask import Flask, send_file
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


class Model(Resource):
    def get(self):
        return send_file("model.pkl", attachment_filename="Model. Version: 0.1")


class ModelOnnx(Resource):
    def get(self):
        return send_file("model.onnx", attachment_filename="Model. Version: 0.1")


class MyModelOnnx(Resource):
    def get(self):
        return send_file("myModel.onnx", attachment_filename="Model. Version: 0.1")


api.add_resource(Model, "/model-pkl")
api.add_resource(MyModelOnnx, "/predict-age-model")
api.add_resource(ModelOnnx, "/model-onnx")

if __name__ == "__main__":
    app.run()

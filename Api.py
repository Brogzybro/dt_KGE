from flask import Flask, request, send_file
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


class Model(Resource):
    def get(self):
        return send_file("model.pkl", attachment_filename="Model. Version: 0.1")


api.add_resource(Model, "/Model")

if __name__ == "__main__":
    app.run()

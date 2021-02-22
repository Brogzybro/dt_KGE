import DataBaseHandler
import urllib.request
import traceback
import json
import requests
url = "http://localhost:3000/certificate/data/"
req = urllib.request.Request(url)

counter = 0


def get_data():
    new_data = []
    try:
        data = requests.get(url,
                            params={'certificateKey' : "60197f7f8447756484fe1d56"})
        jsonArray = data.json()
        for object in jsonArray:
            if 'userId' in object:
                print(object['userId'])

                if DataBaseHandler.is_user_new(object):
                    DataBaseHandler.store_user(object)
            if 'type' in object:
                if DataBaseHandler.is_sample_new(object):
                    if object['value'] > 1000 and object['type'] != 'weight':
                        object['value'] = object['value'] / 1000
                    new_data.append(object)
                    DataBaseHandler.store_sample(object)
        return new_data
    except Exception:
        traceback.print_exc()
        return new_data


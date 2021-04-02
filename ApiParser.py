import DataBaseHandler
import urllib.request
import requests
import logging
import os
from dotenv import load_dotenv


load_dotenv()

url = os.environ.get("DT_BACK_URI")
req = urllib.request.Request(url)

counter = 0


def get_data():
    new_data = []
    try:
        data = requests.get(url,
                            params={'certificateKey' : "60197f7f8447756484fe1d56"})
        jsonArray = data.json()
        logging.info(jsonArray)
        for object in jsonArray:
            if 'userId' in object:
                logging.info(object['userId'])

                if DataBaseHandler.is_user_new(object):
                    DataBaseHandler.store_user(object)
            if 'type' in object:
                if DataBaseHandler.is_sample_new(object):
                    if object['value'] > 1000 and object['type'] != 'weight':
                        object['value'] = object['value'] / 1000
                    new_data.append(object)
                    DataBaseHandler.store_sample(object)
        return new_data
    except Exception as e:
        logging.error("Error fetching data from DT-back API, url: %s", url, exc_info=e)
        return new_data



import pymongo
from gridfs import GridFS
from bson import objectid

myclient = pymongo.MongoClient("")

knowledgeGeneratorDB = myclient['KnowledgeGenerator']
Users = knowledgeGeneratorDB['user']
Samples = knowledgeGeneratorDB['samples']
Models = knowledgeGeneratorDB['models']
fs = GridFS(knowledgeGeneratorDB)


def store_user(user_obj):
    if is_user_new(user_obj):
        print("New user:")
        print(user_obj)
        Users.insert_one(user_obj)


def store_sample(sample_obj):
    if is_sample_new(sample_obj):
        Samples.insert_one(sample_obj)


def store_model(pickle_file, filename, version, desc):
    with open(pickle_file, 'rb') as file:
        fs.put(file, filename=filename, version=version, description=desc)


def is_user_new(user_obj):
    check = Users.count_documents({"userId": user_obj["userId"]}, limit=1)
    if check == 0:
        return True
    return False


def is_sample_new(sample_obj):
    check = Samples.count_documents({"_id": sample_obj["_id"]}, limit=1)
    if check == 0:
        return True
    return False


def get_samples(type=""):
    query = {}
    if type != "":
        query = {"type": type}
    print("query: ", query)
    result = Samples.find(query)

    return result


def get_corresponding_bp_sample(sample):
    look_for = ""
    group_id = sample["grpid"]
    if sample['type'] == "systolicBloodPressure":
        look_for = "diastolicBloodPressure"
    if sample['type'] == "diastolicBloodPressure":
        look_for = "systolicBloodPressure"
    if look_for == "": return
    query = {"type": look_for, "grpid": group_id}
    sample = Samples.find_one(query)
    return sample


def get_user(user_id):
    result = Users.find_one({"userId": user_id})
    return result


def delete_all_users():
    Users.delete_many({})


def delete_all_samples():
    Samples.delete_many({})

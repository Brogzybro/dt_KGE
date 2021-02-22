import sched, time

from sklearn.preprocessing import MinMaxScaler
import pickle
import DataBaseHandler
import ApiParser
import random
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
from sklearn import preprocessing

s = sched.scheduler(time.time, time.sleep)

beta_0 = -15.139611
beta_1 = 0.048337
beta_2 = 0.055844
beta_3 = 0.060932
age = 23
sys = 128
dia = 80

z = beta_0 + age* beta_1 + sys*beta_2 + dia*beta_3
y = 1/ (1 + np.exp(-z))
print(z)

df = pd.read_csv('testData.csv')


def train_model(df):
    print("Traingin Model.")
    print(df)
    scaler = preprocessing.StandardScaler()
    df[['sys', 'dia', 'age']] = scaler.fit_transform(df[['sys', 'dia', 'age']])
    X = df[['sys', 'dia']]
    Y = df['age']

    sb.kdeplot(df['sys'])
    sb.kdeplot(df['dia'])
    sb.kdeplot(df['age'])
    sb.swarmplot(data=df)
    plt.show()
    #model.fit(X.values, Y.values)
    print("\n model Train Iteration")
    #print(model.intercept_)
    #print(model.coef_)


def tune_model(df):
    print("Tuning Model.")
    print(df)
    scaler = preprocessing.StandardScaler()
    df[['sys', 'dia', 'age']] = scaler.fit_transform(df[['sys', 'dia', 'age']])
    X = df[['sys', 'dia']]
    Y = df['age']

    #model.partial_fit(X, Y, classes=np.unique(Y))
    print("\n model Tune Iteration")
    #print(model.intercept_)
    #print(model.coef_)


dataFrame = pd.DataFrame()
dataFrame.at[0,'age'] = beta_1
dataFrame.at[0,'sys'] = beta_2
dataFrame.at[0,'dia'] = beta_3

print("DataFrame: ")
print(dataFrame)


sample = pd.read_csv('sample.csv')


def add_sample_to_df(sample_obj, df):
    user_id = sample_obj['user']
    user = DataBaseHandler.get_user(user_id)
    corresponding_sample = DataBaseHandler.get_corresponding_bp_sample(sample_obj)
    new_data = pd.DataFrame()
    new_data.at[0, 'age'] = user['age']
    new_data.at[0, 'sys'] = sample_obj['value']
    new_data.at[0, 'dia'] = corresponding_sample['value']
    df = df.append(new_data, ignore_index=True)
    return df


def send_api_request():
    print("Checking DT-API for data.")
    new_samples = ApiParser.get_data()
    if len(new_samples) > 0:
        print("New samples found.")
        new_df = pd.DataFrame()
        for s in new_samples:
            if s['type'] == "systolicBloodPressure":
                new_df = add_sample_to_df(s, new_df)
        print("DataFrame from new Samples:")
        print(new_df)
        tune_model(new_df)
    else:
        print("No new samples found.")


def save_model(model):
    pickle.dump(model, open("model.pkl", "wb"))


def load_model(filename):
    model = pickle.load(open(filename, "rb"))
    return model


def get_all_systolic_samples():
    samples = DataBaseHandler.get_samples(type="systolicBloodPressure")
    new_df = pd.DataFrame()
    for samp in samples:
        new_df = add_sample_to_df(samp, new_df)
    return new_df


model = SGDClassifier()

print("df Before: ")
print(df)
count = 0
train_model(get_all_systolic_samples())

while True:
    time.sleep(20)
    send_api_request()



print("SGD model:")
save_model(model)
DataBaseHandler.store_model("model.pkl")


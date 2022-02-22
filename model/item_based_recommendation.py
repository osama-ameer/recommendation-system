from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os
from surprise.model_selection import train_test_split
from collections import defaultdict
import pandas as pd

import pickle

# dataset link: https://drive.google.com/file/d/1ACmDVt3IglXv52K40YFbSmtKtpKPf9lK/view?usp=sharing
df = pd.read_csv('ratings_Electronics.csv', names=['userId', 'productId','rating','timestamp'])

df.drop_duplicates(inplace=True)
df.drop(columns=["timestamp"], inplace=True)

# take 10 % of the data
df = df[:int(len(df) * .1)]

# Keep the users where the user has rated more than 50 
counts1 = df['userId'].value_counts()
Data_new = df[df['userId'].isin(counts1[counts1 >= 50].index)]

# Reading the dataset
reader = Reader(rating_scale=(1, 5))
data1 = Dataset.load_from_df(Data_new,reader)
data1


# Splitting the dataset
trainset, testset = train_test_split(data1, test_size=0.3,random_state=123)

# Use user_based true/false to switch between user-based or item-based collaborative filtering
item_based_algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
item_based_algo.fit(trainset)

item_based_algo.trainset = trainset
item_based_algo.testset = testset

filename = 'item_based_model.sav'
pickle.dump(item_based_algo, open(filename, 'wb'))    # Save model



user_id = "A3OXHLG6DIBRW8"
n_item = 10

# Load Model and Predict
filename = 'item_based_model.sav'
item_based_algo = pickle.load(open(filename, 'rb'))     # load model


# run the trained model against the testset
test_pred = item_based_algo.test(item_based_algo.testset)


def get_top_n(predictions, n=5):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


top_n = get_top_n(test_pred, n=n_item)

# print(top_n.keys()) # User id list

print(top_n[user_id])
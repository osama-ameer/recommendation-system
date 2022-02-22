from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
import pickle
from collections import defaultdict

# Create your views here.

class UserBased(APIView):
    def get(self, request, user_id, *args, **kwargs):
        filename = 'model/user_based_model.sav'
        user_based_algo = pickle.load(open(filename, 'rb'))     # load model

        # run the trained model against the testset
        test_pred = user_based_algo.test(user_based_algo.testset)

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
        top_n = get_top_n(test_pred, n=5)
        # print(top_n.keys()) # User id list
        # print(top_n[user_id])
        return Response(data=top_n[user_id])


class ItemBased(APIView):
    def get(self, request, user_id, *args, **kwargs):
        filename = 'model/item_based_model.sav'
        item_based_algo = pickle.load(open(filename, 'rb'))
        test_pred = item_based_algo.test(item_based_algo.testset)
        def get_top_n(predictions, n=5):
            top_n = defaultdict(list)
            for uid, iid, true_r, est, _ in predictions:
                top_n[uid].append((iid, est))
            for uid, user_ratings in top_n.items():
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                top_n[uid] = user_ratings[:n]
            return top_n
        top_n = get_top_n(test_pred, n=5)
        return Response(data=top_n[user_id])
from django.urls import path, include

from app import views

urlpatterns = [
    # path("", views.Dashboard.as_view(), name="dashboard"),
    path("userBased/<user_id>/", views.UserBased.as_view(), name="user_based"),
    path("itemBased/<user_id>/", views.ItemBased.as_view(), name="item_based"),
]
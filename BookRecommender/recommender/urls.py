from django.urls import path

from . import views


urlpatterns = [
     path('', views.index, name='app.html'),
#     path('', HomePageView.as_view(), name='home-view'),
]
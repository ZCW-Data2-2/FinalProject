from django.urls import path

from . import views


urlpatterns = [
     path('', views.index, name='app.html'),
     path('search_books', views.search_books, name='search-books'),
#     path('', HomePageView.as_view(), name='home-view'),
]


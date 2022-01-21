from django.urls import path

from . import views
from .views import HomePageView, AboutPageView

urlpatterns = [
     path('', HomePageView.as_view(), name='home'),
     path('about/', AboutPageView.as_view(), name='about'),
     path('search_books', views.search_books, name='search-books'),
     path('app', views.app, name='app'),
]


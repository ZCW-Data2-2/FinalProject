from django.shortcuts import render
from django.http import HttpResponse
import requests

from django.views.generic import TemplateView
from .models import recommender_book

from .engine import runEngine

# class HomePageView(TemplateView):
#     template_name = '_base.html'

# placeholder
def index(request):
    return HttpResponse("Hello, would you like to read a book?")

def search_books(request):
    if request.method == "POST":
        search_ui = request.POST['searched']
        search_ui = int(search_ui)
        runEngine(search_ui)
        books = recommender_book.objects.filter(UserID=search_ui)
        return render(request, 'search_books.html',
                        {'searched': search_ui,
                         'books': books})
    else:
        return render(request, 'search_books.html', {})




# class HomePageView(TemplateView):
#     template_name = 'recommender/app.html'

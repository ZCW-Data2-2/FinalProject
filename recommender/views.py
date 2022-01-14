from django.shortcuts import render
from django.http import HttpResponse
import requests

from django.views.generic import TemplateView
from .models import Book

# class HomePageView(TemplateView):
#     template_name = '_base.html'

# placeholder
def index(request):
    return HttpResponse("Hello, would you like to read a book?")

def search_books(request):
    if request.method == "POST":
        searched = request.POST['searched']
        books = Book.objects.filter(name__contains=searched)
        return render(request, 'search_books.html',
                        {'searched': searched,
                         'books': books})
    else:
        return render(request, 'search_books.html', {})




# class HomePageView(TemplateView):
#     template_name = 'recommender/app.html'

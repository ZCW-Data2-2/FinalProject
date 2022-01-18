from django.shortcuts import render
from django.http import HttpResponse
import requests

from django.views.generic import TemplateView
from .models import recommender_book

class HomePageView(TemplateView):
    template_name = 'home.html'

class AboutPageView(TemplateView):
    template_name = 'about.html'

# placeholder
def index(request):
    return HttpResponse("Hello, would you like to read a book?")

def search_books(request):
    if request.method == "POST":
        searched = request.POST['searched']
        books = recommender_book.objects.filter(UserID=searched)
        return render(request, 'search_books.html',
                        {'searched': searched,
                         'books': books})
    else:
        return render(request, 'search_books.html', {})



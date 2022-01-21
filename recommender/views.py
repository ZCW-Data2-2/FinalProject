from django.shortcuts import render
from django.http import HttpResponse
import requests

from django.views.generic import TemplateView
from .models import recommender_book

from .engine import runEngine
from .coldengine import coldrunEngine

class HomePageView(TemplateView):
    template_name = 'home.html'

class AboutPageView(TemplateView):
    template_name = 'about.html'

# class AppPageView(TemplateView):
#     template_name = 'app.html'

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



def app(request):
    if request.method == "POST":
        humor = request.POST['humor']
        horror = request.POST['horror']
        romance = request.POST['romance']
        thriller = request.POST['thriller']
        nonfiction = request.POST['nonfiction']
        hp = request.POST['hp']
        # search_ui = int(search_ui[1])
        # print(humor)
        # print(horror)
        # print(romance)
        # print(thriller)
        # print(nonfiction)
        coldrunEngine(humor, horror, romance, thriller, nonfiction, hp)
        books = recommender_book.objects.filter(UserID=1)
        return render(request, 'app.html',
                        {'searched': f"Humor: {humor}\n Horror: {horror}\n Romance: {romance}\n Thriller: {thriller}\n Non-Fiction: {nonfiction}\n Harry Potter Books: {hp}",
                         'books': books})
    else:
        return render(request, 'app.html', {})



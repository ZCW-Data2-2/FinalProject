from django.shortcuts import render,redirect
from django.http import HttpResponse
import requests
# from .models import cold_start
# from .forms import ColdStartForm

from django.views.generic import TemplateView
from .models import recommender_book

from .engine import runEngine

# class HomePageView(TemplateView):
#     template_name = '_base.html'

# placeholder
def index(request):
    return HttpResponse("Hello, would you like to read a book?")


# def form_view(request):
#     form = ColdStartForm()
#     if request.method == 'POST':
#         form = ColdStartForm(request.POST)
#         if form.is_valid():
#             form.save()
#         return redirect('/')
#     return render(request,'star/Home.html',{'form':form})


def search_books(request):
    if request.method == "POST":
        humor = request.POST['humor']
        horror = request.POST['horror']
        romance = request.POST['romance']
        thriller = request.POST['thriller']
        nonfiction = request.POST['nonfiction']
        # search_ui = int(search_ui[1])
        # print(humor)
        # print(horror)
        # print(romance)
        # print(thriller)
        # print(nonfiction)
        runEngine(humor, horror, romance, thriller, nonfiction)
        books = recommender_book.objects.filter(UserID=1)
        return render(request, 'search_books.html',
                        {'searched': "Here you go!",
                         'books': books})
    else:
        return render(request, 'search_books.html', {})

# def cold_input(request):
#     if request.method == "POST":

#         cold = request.POST['fantasy']
#         print(cold)
#         s = int(search_ui)
#         runEngine(search_ui)
#         books = recommender_book.objects.filter(UserID=search_ui)
#         return render(request, 'search_books.html',
#                         {'searched': search_ui,
#                          'books': books})
#     else:
#         return render(request, 'search_books.html', {})

# cold_input(5)
# class HomePageView(TemplateView):
#     template_name = 'recommender/app.html'

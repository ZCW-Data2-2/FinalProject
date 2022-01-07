from django.shortcuts import render
from django.http import HttpResponse
import requests

from django.views.generic import TemplateView


# class HomePageView(TemplateView):
#     template_name = '_base.html'

# placeholder
def index(request):
    return HttpResponse("Hello, would you like to read a book?")

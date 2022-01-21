from django.contrib import admin
from .models import recommender_book

class recommender_bookAdmin(admin.ModelAdmin):
    pass
admin.site.register(recommender_book, recommender_bookAdmin)
# Register your models here.

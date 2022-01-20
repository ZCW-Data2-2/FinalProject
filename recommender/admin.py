from django.contrib import admin

# Register your models here.
from recommender.models import recommender_book, cold_start

class recommender_bookAdmin(admin.ModelAdmin):
    pass

class cold_startAdmin(admin.ModelAdmin):
    pass

admin.site.register(recommender_book, recommender_bookAdmin)
admin.site.register(cold_start, cold_startAdmin)
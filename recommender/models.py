from django.db import models

# Create your models here.

# We need a model for book table (EC)
class recommender_book(models.Model):
    # name=models.CharField(max_length=200)
    UserID=models.IntegerField()
    BookTitle=models.CharField(max_length=200)
    BookRating=models.IntegerField()
    
    def __str__(self):
        return self.BookTitle

class cold_start(models.Model):
    UserID=models.IntegerField()
    Thrillers=models.IntegerField()
    Romance=models.IntegerField()
    Nonfiction=models.IntegerField()
    Humor=models.IntegerField()
    Horror=models.IntegerField()

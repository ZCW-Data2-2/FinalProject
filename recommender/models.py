from django.db import models

# Create your models here.

# We need a model for book table (EC)
class Book(models.Model):
    name=models.CharField(max_length=200)

    def __str__(self):
        return self.name
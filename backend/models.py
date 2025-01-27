from django.db import models;
from django.contrib.auth.models import User;

class UserProfile(models.Model):
    fullName = models.CharField(max_length=255)
    username = models.CharField(max_length=100, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)
    height = models.FloatField()
    weight = models.FloatField()
    gender = models.CharField(max_length=10)

    def __str__(self):
        return self.username




class History(models.Model):
    label = models.CharField(max_length=100)
    model = models.CharField(max_length=50)
    nutritional_info = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE) 


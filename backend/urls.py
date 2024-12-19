from django.urls import path
from .views import predict_food  # Import the view
from .views import RegisterView
from .views import LoginView


urlpatterns = [
    path('predict_food/', predict_food.as_view(), name='predict_food'),  # Define the URL
    path('register/', RegisterView.as_view(), name='register'),  # Define the URL
    path('login/', LoginView.as_view(), name='login'),  # Define the URL
    
]


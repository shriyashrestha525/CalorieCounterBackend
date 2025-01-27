from django.urls import path
from .views import predict_food  # Import the view
from .views import RegisterView
from .views import LoginView
from .views import UserProfileView
from .views import RecentFoodsView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('predict_food/', predict_food.as_view(), name='predict_food'),  # Define the URL
    path('register/', RegisterView.as_view(), name='register'),  # Define the URL
    path('login/', LoginView.as_view(), name='login'),  # Define the URL
    path('user-profile/', UserProfileView.as_view(), name='user-profile'),
    path('recent_foods/', RecentFoodsView.as_view(), name='recent_foods'),
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    

]


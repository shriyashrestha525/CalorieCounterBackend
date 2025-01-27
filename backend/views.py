from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from django.conf import settings
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import pandas as pd
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
import sys
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.serializers import ValidationError
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth import authenticate
from rest_framework.parsers import MultiPartParser, FormParser
from .models import UserProfile
from .serializers import UserProfileSerializer
from django.contrib.auth.hashers import make_password
from .models import History
from django.utils.timezone import now, make_aware
from datetime import datetime

class UserProfileView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        # Get the data from the frontend (request.data contains the POST data)
        print(request.data)
        data = request.data
        data['password'] = make_password(data['password'])  # Hash the password before saving


        serializer = UserProfileSerializer(data=data)

        if serializer.is_valid():
            serializer.save()  # Save the new user profile to the database
            return Response({'message': 'User profile created successfully'}, status=status.HTTP_201_CREATED)
        else:
            print(serializer.errors)  # Log the serializer errors to understand the issue
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        #return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        

        try:
            user_profile = UserProfile.objects.get(username=request.user.username)
            serializer = UserProfileSerializer(user_profile)
            return Response(serializer.data)
        except UserProfile.DoesNotExist:
            return Response({'error': 'User profile not found'}, status=status.HTTP_404_NOT_FOUND)
        


class RecentFoodsView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            # Get today's date
            today = now().date()
            start_of_day = make_aware(datetime.combine(today, datetime.min.time()))
            end_of_day = make_aware(datetime.combine(today, datetime.max.time()))

            
            # Fetch only today's food records
            recent_foods = History.objects.filter(
                user=request.user,
                created_at__range=(start_of_day, end_of_day))

            data = [
                {
                    'label': food.label,
                    'model': food.model,
                    'nutritional_info': food.nutritional_info,
                }
                for food in recent_foods
            ]

            return Response({'recent_foods': data})
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class RegisterView(APIView):
    permission_classes = [AllowAny] 

    def post(self, request):
        username = request.data.get('username')
        email = request.data.get('email')
        password = request.data.get('password')

        errors = {}

        try:
            validate_email(email)
        except ValidationError:
            errors['email'] = ['Invalid email format']

        if User.objects.filter(username=username).exists():
            errors['username'] = ['Username already exists']
        
        if User.objects.filter(email=email).exists():
            if 'email' not in errors:
                errors['email'] = []
            errors['email'].append('Email already exists')

        try:
            validate_password(password)
        except ValidationError as e:
            errors['password'] = e.messages

        if errors:
            return Response({'errors': errors}, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
        )

        user.save()

        
        
        return Response({
            'message': 'User registered successfully',
        }, status=status.HTTP_201_CREATED)
    
class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        print("Request Data:", request.data)


    

        # Authenticate the user
        user = authenticate(username=username, password=password)

        if user is not None:
            # Generate JWT tokens
            refresh = RefreshToken.for_user(user)
            access_token = str(refresh.access_token)

            return Response({
                'message': 'Login successful',
                'access_token': access_token,
                
            }, status=status.HTTP_200_OK)
        else:
            if not User.objects.filter(username=username).exists():
                return Response({'error': 'Invalid username'}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({'error': 'Invalid password'}, status=status.HTTP_400_BAD_REQUEST)
            
            

sys.path.append("../food_recognition")
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'food_recognition.settings')
# Load model and nutrition data
model_path = os.path.join(settings.BASE_DIR, 'backend/models', 'FV.h5')
model_path_mobilenet = os.path.join(settings.BASE_DIR, 'backend/models', 'mobilenet.h5')
model_path_inception = os.path.join(settings.BASE_DIR, 'backend/models', 'inceptionV3.h5')

model = load_model(model_path)
model_mobilenet = load_model(model_path_mobilenet)
model_inception = load_model(model_path_inception)

nutrition_data = pd.read_csv(os.path.join(settings.BASE_DIR, 'nutrition101.csv'))

LABELS = ['apple pie',
         'baby back ribs',
         'baklava',
         'beef carpaccio',
         'beef tartare',
         'beet salad',
         'beignets',
         'bibimbap',
         'bread pudding',
         'breakfast burrito',
         'bruschetta',
         'caesar salad',
         'cannoli',
         'caprese salad',
         'carrot cake',
         'ceviche',
         'cheese plate',
         'cheesecake',
         'chicken curry',
         'chicken quesadilla',
         'chicken wings',
         'chocolate cake',
         'chocolate mousse',
         'churros',
         'clam chowder',
         'club sandwich',
         'crab cakes',
         'creme brulee',
         'croque madame',
         'cup cakes',
         'deviled eggs',
         'donuts',
         'dumplings',
         'edamame',
         'eggs benedict',
         'escargots',
         'falafel',
         'filet mignon',
         'fish and_chips',
         'foie gras',
         'french fries',
         'french onion soup',
         'french toast',
         'fried calamari',
         'fried rice',
         'frozen yogurt',
         'garlic bread',
         'gnocchi',
         'greek salad',
         'grilled cheese sandwich',
         'grilled salmon',
         'guacamole',
         'gyoza',
         'hamburger',
         'hot and sour soup',
         'hot dog',
         'huevos rancheros',
         'hummus',
         'ice cream',
         'lasagna',
         'lobster bisque',
         'lobster roll sandwich',
         'macaroni and cheese',
         'macarons',
         'miso soup',
         'mussels',
         'nachos',
         'omelette',
         'onion rings',
         'oysters',
         'pad thai',
         'paella',
         'pancakes',
         'panna cotta',
         'peking duck',
         'pho',
         'pizza',
         'pork chop',
         'poutine',
         'prime rib',
         'pulled pork sandwich',
         'ramen',
         'ravioli',
         'red velvet cake',
         'risotto',
         'samosa',
         'sashimi',
         'scallops',
         'seaweed salad',
         'shrimp and grits',
         'spaghetti bolognese',
         'spaghetti carbonara',
         'spring rolls',
         'steak',
         'strawberry shortcake',
         'sushi',
         'tacos',
         'octopus balls',
         'tiramisu',
         'tuna tartare',
         'waffles']


import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# image_path = 'momo.jpg'
# img = load_img(image_path, target_size=(224, 224))
# img_array = img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array = img_array / 255.0


# predictions = model.predict(img_array)


# predicted_index = np.argmax(predictions)
# predicted_label = LABELS[predicted_index]
# print(predicted_label)


# food_info = nutrition_data[nutrition_data['name'] == predicted_label]
# calories = food_info['calories']
# protein = food_info['protein']
# calcium = food_info['calcium']
# carbs = food_info['carbohydrates']
# fat = food_info['fat']

# print(f"Estimated Calories: {calories}")
# print(f"Estimated Protein: {protein}")
# print(f"Estimated calcium: {calcium}")
# print(f"Estimated carbohydrates: {carbs}")
# print(f"Estimated fat: {fat}")




class predict_food(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        if request.method == 'POST' and request.FILES['image']:
            if 'image' not in request.FILES:
                return JsonResponse({'error': 'No image file uploaded'}, status=400)
            image = request.FILES['image']
            model_type = request.POST.get('model_type', 'cnn')

            try:
                temp_file_path = default_storage.save(f'temp/{image.name}', ContentFile(image.read()))

                temp_file_path_full = os.path.join(settings.MEDIA_ROOT, temp_file_path)
                img = load_img(temp_file_path_full, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                if model_type == 'mobilenet':
                    predictions = model_mobilenet.predict(img_array)
                    predicted_index = np.argmax(predictions)
                    predicted_label = LABELS[predicted_index]
                    print(predicted_label)
                
                elif model_type == 'inception':
                    predictions = model_inception.predict(img_array)
                    predicted_index = np.argmax(predictions)
                    predicted_label = LABELS[predicted_index]
                    print(predicted_label)

                else:
                    predictions = model_mobilenet.predict(img_array)
                    predicted_index = np.argmax(predictions)
                    predicted_label = LABELS[predicted_index]
                    print(predicted_label)


                food_info = nutrition_data[nutrition_data['name'] == predicted_label]
                food_info_dict = food_info.iloc[0].to_dict()

                # Convert each value to a basic Python type (str, int, or float)
                for key, value in food_info_dict.items():
                    if isinstance(value, (np.int64, np.float64)):
                        food_info_dict[key] = value.item()
                    else:
                        food_info_dict[key] = str(value)  # Ensure string conversion for non-numeric values
                
                portion = food_info_dict.get('portion', 'N/A')
                # Prepare the nutrition table
                nutrition_table = [
                    {'name': 'calories', 'value': food_info_dict.get('calories', 'N/A')},
                    {'name': 'protein', 'value': food_info_dict.get('protein', 'N/A')},
                    {'name': 'fat', 'value': food_info_dict.get('fat', 'N/A')},
                    {'name': 'carbohydrates', 'value': food_info_dict.get('carbohydrates', 'N/A')},
                    
                ]

                user = request.user
                History.objects.create(
                    label=predicted_label,
                    model=model_type,
                    nutritional_info=nutrition_table,
                    user=user
                )
                # calories = food_info['calories']
                # protein = food_info['protein']
                # calcium = food_info['calcium']
                # carbs = food_info['carbohydrates']
                # fat = food_info['fat']

                # print(f"Estimated Calories: {calories}")
                # print(f"Estimated Protein: {protein}")
                # print(f"Estimated calcium: {calcium}")
                # print(f"Estimated carbohydrates: {carbs}")
                # print(f"Estimated fat: {fat}")

                default_storage.delete(temp_file_path)

                
                return JsonResponse({'message': 'Prediction successful', 'result': predicted_label, 'portion': portion, 'Nutrition' : nutrition_table})
            
            except Exception as e:
                return JsonResponse({'error': f'Error processing image: {str(e)}'}, status=500)


        else:
            # If the request method is not POST, return a method not allowed response
            return JsonResponse({'error': 'Invalid request method, only POST allowed'}, status=405)
            
            

        
import os
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
import pickle
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.management import execute_from_command_line
from django.conf import settings
from django.urls import path
from django.shortcuts import HttpResponse
from datetime import datetime

# Constants
MODEL_PATH = "sucrose_model.pkl"
REFERENCE_DATE = pd.to_datetime("2024-01-01")  # Replace with your training reference date
TRAINING_FILE_PATH = r"C:\Users\amodi.s\Desktop\Pooled_OLS\training_1.xlsx"  # Direct path to your training dataset


# Function to train the model
def train_model(training_file):
    print("Training model...")

    # Load training data
    df_train = pd.read_excel(training_file, header=1, names=['Date', 'NDVI', 'NDWI', 'CIG', 'Sucrose'])
    df_train['Date'] = pd.to_datetime(df_train['Date']).dt.tz_localize(None)
    df_train['NDVI'] = pd.to_numeric(df_train['NDVI'])
    df_train['NDWI'] = pd.to_numeric(df_train['NDWI'])
    df_train['CIG'] = pd.to_numeric(df_train['CIG'])
    df_train['Sucrose'] = pd.to_numeric(df_train['Sucrose'])

    # Prepare features for regression
    X_train = df_train[['Date', 'NDVI', 'NDWI', 'CIG']]
    min_date_train = X_train['Date'].min()
    X_train['Date'] = (X_train['Date'] - min_date_train) / np.timedelta64(1, 'D')
    X_train = add_constant(X_train)
    y_train = df_train['Sucrose']

    # Train the model
    model = OLS(y_train, X_train).fit()

    # Save the model
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)
    print(f"Model trained and saved to {MODEL_PATH}")


# Django API View to test the model
@csrf_exempt
def predict_sucrose(request):
    if request.method == "POST":
        # Check if file is provided
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        # Get the uploaded file
        uploaded_file = request.FILES['file']

        # Validate file type
        if not uploaded_file.name.endswith(('.xlsx', '.xls')):
            return JsonResponse({"error": "Uploaded file is not a valid Excel file"}, status=400)

        try:
            # Load the trained model
            if not os.path.exists(MODEL_PATH):
                return JsonResponse({"error": "Model not found. Train the model first."}, status=400)
            with open(MODEL_PATH, "rb") as file:
                model = pickle.load(file)

            # Read the uploaded Excel file
            df_test = pd.read_excel(uploaded_file, engine='openpyxl')
            df_test.columns = df_test.columns.str.strip()

            # Validate required columns
            required_columns = {"Date", "NDVI", "NDWI", "CIG"}
            missing_columns = required_columns - set(df_test.columns)
            if missing_columns:
                return JsonResponse({"error": f"Missing required columns: {list(missing_columns)}"}, status=400)

            # Preprocess test data
            df_test['Date'] = pd.to_datetime(df_test['Date'], errors='coerce').dt.tz_localize(None)
            features = df_test[['Date', 'NDVI', 'NDWI', 'CIG']]
            features['Date'] = (features['Date'] - REFERENCE_DATE) / np.timedelta64(1, 'D')
            features = add_constant(features, has_constant='add')

            # Predict using the model
            predictions = model.predict(features)
            df_test['Predicted Sucrose'] = predictions

            # Return predictions as JSON
            return JsonResponse({"predictions": df_test.to_dict(orient='records')}, status=200)
        except Exception as e:
            return JsonResponse({"error": f"Error processing file: {str(e)}"}, status=400)
    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)


# Django setup for standalone script
if not settings.configured:
    settings.configure(
        DEBUG=True,
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=['*'],
        SECRET_KEY='a_random_secret_key',
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',  # Add this for auth-related functionality
        ],
        MIDDLEWARE=[
            'django.middleware.common.CommonMiddleware',
            'django.middleware.csrf.CsrfViewMiddleware',
            'django.middleware.security.SecurityMiddleware',
            'django.contrib.sessions.middleware.SessionMiddleware',
            'django.contrib.auth.middleware.AuthenticationMiddleware',
            'django.contrib.messages.middleware.MessageMiddleware',
        ],
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.sqlite3',
                'NAME': os.path.join(os.path.dirname(__file__), 'db.sqlite3'),
            }
        },
        TEMPLATES=[
            {
                'BACKEND': 'django.template.backends.django.DjangoTemplates',
                'DIRS': [],
                'APP_DIRS': True,
                'OPTIONS': {
                    'context_processors': [
                        'django.template.context_processors.debug',
                        'django.template.context_processors.request',
                        'django.contrib.auth.context_processors.auth',
                        'django.contrib.messages.context_processors.messages',
                    ],
                },
            },
        ],
    )


urlpatterns = [
    path('', lambda request: HttpResponse("Sucrose Prediction API is running!")),
    path('predict/', predict_sucrose),
]


if __name__ == "__main__":
    # Training the model
    print(f"Training model using {TRAINING_FILE_PATH}...")
    train_model(TRAINING_FILE_PATH)
    # Starting the Django API server
    print("Starting Django API server...")
    execute_from_command_line(['manage.py', 'runserver', '8000'])
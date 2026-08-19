from django.urls import path
from esports import views

urlpatterns = [
    path('',          views.home,        name='home'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('analysis/', views.analysis,    name='analysis'),
    path('teams/',    views.teams_view,  name='teams'),
]

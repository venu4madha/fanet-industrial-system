from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    path('profile/', views.profile_view, name='profile'),
    path('users/', views.admin_user_list, name='user_list'),
    path('users/create/', views.admin_user_create, name='user_create'),
    path('users/<int:user_id>/edit/', views.admin_user_edit, name='user_edit'),
    path('users/<int:user_id>/delete/', views.admin_user_delete, name='user_delete'),
    path('users/<int:user_id>/activity/', views.admin_user_activity, name='user_activity'),
]

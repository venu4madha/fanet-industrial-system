from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils.decorators import method_decorator
from .models import CustomUser
from .forms import LoginForm, UserCreateForm, UserEditForm
from inspection.models import UploadedImage, TrainingSession


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = LoginForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            ip = request.META.get('REMOTE_ADDR')
            CustomUser.objects.filter(pk=user.pk).update(last_login_ip=ip)
            messages.success(request, f'Welcome back, {user.get_full_name() or user.username}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()
    return render(request, 'accounts/login.html', {'form': form})


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = UserCreateForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.username}! Your account has been created successfully.')
            return redirect('dashboard')
        else:
            messages.error(request, 'Registration failed. Please correct the errors below.')
    else:
        form = UserCreateForm()
    return render(request, 'accounts/register.html', {'form': form})


@login_required
def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('accounts:login')


@login_required
def profile_view(request):
    return render(request, 'accounts/profile.html', {'user': request.user})


@login_required
def admin_user_list(request):
    if not request.user.is_admin_role:
        messages.error(request, 'Access denied.')
        return redirect('dashboard')
    users = CustomUser.objects.all().order_by('-created_at')
    return render(request, 'accounts/user_list.html', {'users': users})


@login_required
def admin_user_create(request):
    if not request.user.is_admin_role:
        messages.error(request, 'Access denied.')
        return redirect('dashboard')
    if request.method == 'POST':
        form = UserCreateForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'User created successfully.')
            return redirect('accounts:user_list')
    else:
        form = UserCreateForm()
    return render(request, 'accounts/user_form.html', {'form': form, 'action': 'Create'})


@login_required
def admin_user_edit(request, user_id):
    if not request.user.is_admin_role:
        messages.error(request, 'Access denied.')
        return redirect('dashboard')
    user = get_object_or_404(CustomUser, pk=user_id)
    if request.method == 'POST':
        form = UserEditForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, 'User updated successfully.')
            return redirect('accounts:user_list')
    else:
        form = UserEditForm(instance=user)
    return render(request, 'accounts/user_form.html', {'form': form, 'action': 'Edit', 'edit_user': user})


@login_required
def admin_user_delete(request, user_id):
    if not request.user.is_admin_role:
        messages.error(request, 'Access denied.')
        return redirect('dashboard')
    user = get_object_or_404(CustomUser, pk=user_id)
    if request.method == 'POST':
        if user == request.user:
            messages.error(request, 'You cannot delete your own account.')
        else:
            username = user.username
            user.delete()
            messages.success(request, f'User "{username}" deleted.')
        return redirect('accounts:user_list')
    return render(request, 'accounts/user_confirm_delete.html', {'edit_user': user})


@login_required
def admin_user_activity(request, user_id):
    if not request.user.is_admin_role:
        messages.error(request, 'Access denied.')
        return redirect('dashboard')
    user = get_object_or_404(CustomUser, pk=user_id)
    images = UploadedImage.objects.filter(uploaded_by=user).order_by('-uploaded_at')[:20]
    sessions = TrainingSession.objects.filter(started_by=user).order_by('-started_at')[:10]
    return render(request, 'accounts/user_activity.html', {
        'edit_user': user, 'images': images, 'sessions': sessions
    })

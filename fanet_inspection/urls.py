from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from inspection import views as inspection_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', inspection_views.landing, name='landing'),
    path('dashboard/', inspection_views.dashboard, name='dashboard'),
    path('accounts/', include('accounts.urls')),
    path('inspection/', include('inspection.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

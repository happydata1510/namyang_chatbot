"""
URL configuration for namyangju_SP project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static
from namyangju_SP import views

urlpatterns = [
    path("", views.index, name="index"),
    path("child/", views.child, name="child"),
    path("common/", views.common, name="common"),
    path("elder/", views.elder, name="elder"),
    path("family/", views.family, name="family"),
    path("school/", views.school, name="school"),
    path("sex/", views.sex, name="sex"),
    path("stalking/", views.stalking, name="stalking"),
    path("agency/", views.agency, name="agency"),
    path("api/record_link_click/", views.record_link_click, name="record_link_click"),
    path("api/chatbot/", views.chatbot_api, name="chatbot_api"),
    path("api/chatbot/stream/", views.chatbot_stream_api, name="chatbot_stream_api"),
    path("api/init_knowledge/", views.init_knowledge_base, name="init_knowledge_base"),
    path("chat/", TemplateView.as_view(template_name="chatbot.html")),
]

# 개발 환경에서 정적 파일 서빙
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

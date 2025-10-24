#!/usr/bin/env python
"""
Railway용 간단한 Django 시작 스크립트
"""
import os
import sys
import django
from django.core.management import execute_from_command_line

if __name__ == "__main__":
    # 환경변수 설정
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "namyangju_SP.settings")
    
    # Django 설정
    django.setup()
    
    # 데이터베이스 마이그레이션
    print("Running migrations...")
    execute_from_command_line(["manage.py", "migrate", "--noinput"])
    
    # 정적 파일 수집
    print("Collecting static files...")
    execute_from_command_line(["manage.py", "collectstatic", "--noinput", "--clear"])
    
    # 서버 시작
    print("Starting Django server...")
    port = os.environ.get("PORT", "8000")
    execute_from_command_line(["manage.py", "runserver", f"0.0.0.0:{port}"])

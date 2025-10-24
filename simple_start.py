#!/usr/bin/env python
"""
Railway용 간단한 Django 시작 스크립트
"""
import os
import sys

# 환경변수 설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "namyangju_SP.settings")

# Django import 및 설정
try:
    import django
    from django.core.management import execute_from_command_line
    from django.core.wsgi import get_wsgi_application
    
    print("Django imported successfully")
    
    # Django 설정
    django.setup()
    print("Django setup completed")
    
    # 데이터베이스 마이그레이션
    print("Running migrations...")
    execute_from_command_line(["manage.py", "migrate", "--noinput"])
    print("Migrations completed")
    
    # 정적 파일 수집
    print("Collecting static files...")
    execute_from_command_line(["manage.py", "collectstatic", "--noinput", "--clear"])
    print("Static files collected")
    
    # 서버 시작
    print("Starting Django server...")
    port = os.environ.get("PORT", "8000")
    print(f"Using port: {port}")
    
    # WSGI 애플리케이션으로 시작
    application = get_wsgi_application()
    
    # 간단한 HTTP 서버로 시작
    from wsgiref.simple_server import make_server
    with make_server('0.0.0.0', int(port), application) as httpd:
        print(f"Server running on port {port}")
        httpd.serve_forever()
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

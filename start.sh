#!/bin/bash

# Railway 시작 스크립트
echo "Starting Django application on Railway..."

# 환경변수 확인
echo "PORT: $PORT"
echo "SECRET_KEY: ${SECRET_KEY:0:10}..."
echo "DEBUG: $DEBUG"

# 데이터베이스 마이그레이션 실행
echo "Running database migrations..."
python manage.py migrate --noinput

# 정적 파일 수집
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Django 개발 서버로 시작 (디버깅용)
echo "Starting Django development server..."
python manage.py runserver 0.0.0.0:$PORT

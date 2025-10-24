#!/bin/bash

# Railway 시작 스크립트
echo "Starting Django application on Railway..."

# 데이터베이스 마이그레이션 실행
echo "Running database migrations..."
python manage.py migrate --noinput

# 정적 파일 수집
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Gunicorn으로 애플리케이션 시작
echo "Starting Gunicorn server..."
exec gunicorn namyangju_SP.wsgi:application \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -

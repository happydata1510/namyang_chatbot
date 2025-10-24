# Railway 환경변수 설정 가이드

## 필수 환경변수

Railway 대시보드에서 다음 환경변수들을 설정해야 합니다:

### 1. Django 설정
```
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=*
```

### 2. OpenAI API 설정
```
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Railway 환경 감지
```
RAILWAY_ENVIRONMENT=true
```

## 설정 방법

1. Railway 대시보드 접속
2. 프로젝트 선택
3. 서비스 선택
4. **Variables** 탭 클릭
5. **New Variable** 버튼 클릭
6. 위의 환경변수들을 하나씩 추가

## 보안 주의사항

- SECRET_KEY는 강력한 랜덤 문자열 사용
- OPENAI_API_KEY는 실제 OpenAI API 키 사용
- DEBUG는 프로덕션에서는 False로 설정

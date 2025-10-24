# 남양주남부경찰서 RAG 기반 AI 챗봇

## 프로젝트 개요

이 프로젝트는 남양주남부경찰서를 위한 RAG(Retrieval-Augmented Generation) 기반 AI 챗봇 시스템입니다. ChatGPT API와 벡터 데이터베이스를 활용하여 경찰서 관련 업무에 특화된 지능형 상담 서비스를 제공합니다.

## 주요 기능

- **RAG 기반 답변 생성**: 경찰서 관련 지식 베이스를 기반으로 정확한 답변 제공
- **현대적인 UI/UX**: 반응형 디자인과 세련된 인터페이스
- **다양한 업무 지원**: 신고접수, 교통사고, 분실물, 범죄신고 등 다양한 경찰 업무 안내
- **실시간 상담**: 24시간 AI 챗봇을 통한 즉시 상담 서비스

## 기술 스택

- **Backend**: Django 5.1.6
- **Database**: PostgreSQL
- **AI/ML**: OpenAI GPT-4, Sentence Transformers, ChromaDB
- **Frontend**: HTML5, CSS3, JavaScript, Font Awesome
- **RAG Framework**: LangChain

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`env.example` 파일을 참고하여 `.env` 파일을 생성하고 필요한 환경 변수를 설정하세요:

```bash
cp env.example .env
```

`.env` 파일에서 다음 값들을 실제 값으로 변경하세요:
- `SECRET_KEY`: Django 시크릿 키
- `OPENAI_API_KEY`: OpenAI API 키
- 기타 필요한 환경 변수들

### 3. 데이터베이스 마이그레이션

```bash
python manage.py migrate
```

### 4. RAG 시스템 초기화

```bash
python manage.py init_rag
```

### 5. 서버 실행

```bash
python manage.py runserver
```

## 사용법

### 웹 인터페이스

1. 브라우저에서 `http://localhost:8000` 접속
2. 메인 페이지에서 "AI 챗봇" 카드 클릭
3. 챗봇 페이지에서 "지식 베이스 초기화" 버튼 클릭 (최초 1회)
4. 질문 입력 후 엔터 또는 전송 버튼 클릭

### API 엔드포인트

#### 챗봇 API
- **URL**: `/api/chatbot/`
- **Method**: POST
- **Body**: `{"message": "질문 내용"}`

#### 지식 베이스 초기화 API
- **URL**: `/api/init_knowledge/`
- **Method**: POST

## 프로젝트 구조

```
namyangju_SP/
├── namyangju_SP/
│   ├── management/
│   │   └── commands/
│   │       └── init_rag.py          # RAG 초기화 명령어
│   ├── rag_system.py                # RAG 시스템 구현
│   ├── knowledge_base.py            # 지식 베이스 데이터
│   ├── models.py                    # Django 모델
│   ├── views.py                     # 뷰 함수
│   ├── urls.py                      # URL 패턴
│   └── settings.py                  # 설정
├── templates/
│   ├── index.html                   # 메인 페이지
│   ├── chatbot.html                 # 챗봇 페이지
│   └── ...
├── static/
│   ├── css/
│   │   └── style.css                # 스타일시트
│   └── js/
│       └── script.js                # JavaScript
├── requirements.txt                 # 의존성 목록
└── README.md                        # 프로젝트 문서
```

## 지식 베이스

현재 시스템에는 다음과 같은 경찰서 관련 지식이 포함되어 있습니다:

- 기본 정보 (주소, 연락처, 관할구역)
- 신고 및 응급상황 대응
- 교통사고 처리 절차
- 분실물 신고 및 찾기
- 범죄신고 및 수사
- 생활안전 관련 서비스
- 교통관리 및 단속
- 민원서비스 안내
- 사이버범죄 신고 및 대응
- 가정폭력 및 성폭력 신고
- 청소년 관련 범죄 대응
- 마약 및 약물범죄 신고
- 도난 및 절도 신고
- 사기 및 금융범죄 신고
- 응급상황 및 재난대응

## 개발자 정보

- **개발**: AI Assistant
- **프로젝트**: 남양주남부경찰서 RAG 기반 AI 챗봇
- **버전**: 2.0

## 라이선스

이 프로젝트는 남양주남부경찰서와 (주)사이트큐빅의 협력 프로젝트입니다.

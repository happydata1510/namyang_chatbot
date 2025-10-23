from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import LinkClick
from .rag_system import get_rag_system
from .knowledge_base import get_police_knowledge_base
import json
import requests
import logging
import time

logger = logging.getLogger(__name__)


def index(request):
    return render(request, "index.html")


def child(request):
    return render(request, "child.html")


def common(request):
    return render(request, "common.html")


def elder(request):
    return render(request, "elder.html")


def family(request):
    return render(request, "family.html")


def school(request):
    return render(request, "school.html")


def sex(request):
    return render(request, "sex.html")


def stalking(request):
    return render(request, "stalking.html")


def pocketbook(request):
    return render(request, "pocketbook.html")


def agency(request):
    return render(request, "agency.html")


# --- 링크 클릭 기록용 API ---
@csrf_exempt
def record_link_click(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            href = data.get("href")
            if href:
                obj, created = LinkClick.objects.get_or_create(href=href)
                obj.click_count += 1
                obj.save()
                return JsonResponse({"status": "ok", "click_count": obj.click_count})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=400)
    return JsonResponse({"status": "error"}, status=400)


# --- RAG 기반 챗봇 API ---
@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            message = data.get("message", "")

            if not message:
                return JsonResponse(
                    {"status": "error", "message": "메시지가 없습니다."}, status=400
                )

            # OpenAI API 키 확인 (테스트 모드에서는 건너뛰기)
            if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "test-key":
                # 테스트 모드에서는 RAG 시스템의 fallback 응답 사용
                pass

            # RAG 시스템을 통한 응답 생성
            rag_system = get_rag_system()
            bot_response = rag_system.query(message)

            return JsonResponse({"status": "success", "response": bot_response})

        except Exception as e:
            logger.error(f"Error in chatbot_api: {str(e)}")
            return JsonResponse(
                {"status": "error", "message": f"오류가 발생했습니다: {str(e)}"},
                status=500,
            )

    return JsonResponse(
        {"status": "error", "message": "POST 요청만 허용됩니다."}, status=405
    )

# --- 스트리밍 챗봇 API ---
@csrf_exempt
def chatbot_stream_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            message = data.get("message", "")

            if not message:
                return JsonResponse(
                    {"status": "error", "message": "메시지가 없습니다."}, status=400
                )

            def generate_response():
                try:
                    # RAG 시스템을 통한 응답 생성
                    rag_system = get_rag_system()
                    bot_response = rag_system.query(message)
                    
                    # 응답을 문자 단위로 분할하여 스트리밍 (더 자연스러운 타이핑 효과)
                    for i, char in enumerate(bot_response):
                        if i == 0:
                            yield f'data: {{"status": "start", "response": "{char}"}}\n\n'
                        else:
                            yield f'data: {{"status": "streaming", "response": "{char}"}}\n\n'
                        
                        # 문장부호나 공백에서는 더 긴 지연
                        if char in '.!?':
                            time.sleep(0.1)  # 문장 끝에서는 100ms 지연
                        elif char in ',;:':
                            time.sleep(0.05)  # 쉼표에서는 50ms 지연
                        elif char == ' ':
                            time.sleep(0.02)  # 공백에서는 20ms 지연
                        else:
                            time.sleep(0.03)  # 일반 문자에서는 30ms 지연
                    
                    yield f'data: {{"status": "end", "response": ""}}\n\n'
                    
                except Exception as e:
                    logger.error(f"Error in streaming: {str(e)}")
                    yield f'data: {{"status": "error", "message": "오류가 발생했습니다."}}\n\n'

            response = StreamingHttpResponse(
                generate_response(),
                content_type='text/event-stream'
            )
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'  # Nginx 버퍼링 비활성화
            return response

        except Exception as e:
            logger.error(f"Error in chatbot_stream_api: {str(e)}")
            return JsonResponse(
                {"status": "error", "message": f"오류가 발생했습니다: {str(e)}"},
                status=500,
            )

    return JsonResponse(
        {"status": "error", "message": "POST 요청만 허용됩니다."}, status=405
    )


# --- 지식 베이스 초기화 API ---
@csrf_exempt
def init_knowledge_base(request):
    if request.method == "POST":
        try:
            # 지식 베이스 데이터 가져오기
            knowledge_data = get_police_knowledge_base()

            # RAG 시스템에 데이터 추가
            rag_system = get_rag_system()
            success = rag_system.add_documents(knowledge_data)

            if success:
                return JsonResponse(
                    {
                        "status": "success",
                        "message": f"지식 베이스가 성공적으로 초기화되었습니다. ({len(knowledge_data)}개 문서 추가)",
                    }
                )
            else:
                return JsonResponse(
                    {
                        "status": "error",
                        "message": "지식 베이스 초기화에 실패했습니다.",
                    },
                    status=500,
                )

        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            return JsonResponse(
                {"status": "error", "message": f"오류가 발생했습니다: {str(e)}"},
                status=500,
            )

    return JsonResponse(
        {"status": "error", "message": "POST 요청만 허용됩니다."}, status=405
    )

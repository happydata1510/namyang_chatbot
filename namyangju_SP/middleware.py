from .models import VisitLog

class VisitLogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # 제외할 경로 목록
        exclude_paths = ['/favicon.ico', '/api/record_link_click/']
        response = self.get_response(request)
        # 관리 페이지, favicon, API, static 파일 등 제외
        if (
            request.path not in exclude_paths
            and not request.path.startswith('/admin')
            and not request.path.startswith('/static/')
            and not request.path.endswith('.js')
        ):
            VisitLog.objects.create(
                path=request.path,
                user=request.user if request.user.is_authenticated else None,
                ip_address=request.META.get('REMOTE_ADDR')
            )
        return response
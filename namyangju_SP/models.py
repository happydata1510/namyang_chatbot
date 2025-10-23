from django.db import models

class VisitLog(models.Model):
    path = models.CharField(max_length=255)  # 방문한 URL
    timestamp = models.DateTimeField(auto_now_add=True)  # 방문 시간
    user = models.ForeignKey('auth.User', null=True, blank=True, on_delete=models.SET_NULL)  # 로그인 사용자(선택)
    ip_address = models.GenericIPAddressField(null=True, blank=True)  # IP(선택)

    def __str__(self):
        return f"{self.path} at {self.timestamp}"

# href 클릭 기록 및 카운트용 모델
class LinkClick(models.Model):
    href = models.CharField(max_length=500, unique=True)  # 클릭된 링크 URL
    click_count = models.PositiveIntegerField(default=0)  # 클릭 횟수
    last_clicked = models.DateTimeField(auto_now=True)    # 마지막 클릭 시간

    def __str__(self):
        return f"{self.href} ({self.click_count})"
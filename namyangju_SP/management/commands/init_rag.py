"""
RAG 시스템 초기화 명령어
"""

from django.core.management.base import BaseCommand
from namyangju_SP.rag_system import get_rag_system
from namyangju_SP.knowledge_base import get_police_knowledge_base
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "RAG 시스템을 초기화하고 지식 베이스를 구축합니다."

    def handle(self, *args, **options):
        try:
            self.stdout.write("RAG 시스템 초기화를 시작합니다...")

            # RAG 시스템 인스턴스 생성
            rag_system = get_rag_system()

            # 지식 베이스 데이터 가져오기
            knowledge_data = get_police_knowledge_base()

            # 지식 베이스에 데이터 추가
            success = rag_system.add_documents(knowledge_data)

            if success:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"RAG 시스템이 성공적으로 초기화되었습니다. "
                        f"({len(knowledge_data)}개 문서 추가)"
                    )
                )
            else:
                self.stdout.write(self.style.ERROR("RAG 시스템 초기화에 실패했습니다."))

        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            self.stdout.write(self.style.ERROR(f"오류가 발생했습니다: {str(e)}"))

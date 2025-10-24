"""
RAG (Retrieval-Augmented Generation) 시스템 구현
"""

import os
import json
from typing import List, Dict, Any
from django.conf import settings
import logging
import threading
import time
from functools import lru_cache

# 초경량 RAG 시스템 - 기본 라이브러리만 사용
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 기본 라이브러리만 사용
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        
        # 캐시 초기화
        self._response_cache = {}
        self._cache_lock = threading.Lock()
        self._last_cache_cleanup = time.time()

        # 지식베이스 데이터 로드
        self.knowledge_base = []
        self._load_knowledge_base()

        # OpenAI 클라이언트 초기화
        if self.openai_api_key and self.openai_api_key != "test-key" and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"OpenAI client initialization failed: {str(e)}")
                self.openai_client = None
        else:
            self.openai_client = None
            if not self.openai_api_key:
                logger.warning("OpenAI API key not provided")
            elif not OPENAI_AVAILABLE:
                logger.warning("OpenAI package not available")
    
    def _load_knowledge_base(self):
        """지식베이스 데이터 로드"""
        try:
            from namyangju_SP.knowledge_base import get_police_knowledge_base
            self.knowledge_base = get_police_knowledge_base()
            logger.info(f"Loaded {len(self.knowledge_base)} knowledge base documents")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            self.knowledge_base = []
    
    def _cleanup_cache(self):
        """초경량 캐시 정리 (간단한 LRU 방식)"""
        # 캐시 크기 제한으로 자동 정리됨 (query 메서드에서 처리)
        pass

    def add_documents(self, documents: List[Dict[str, Any]]):
        """문서를 지식베이스에 추가"""
        try:
            self.knowledge_base.extend(documents)
            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

    def search_similar_documents(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """초경량 문서 검색 (키워드 기반)"""
        try:
            if not self.knowledge_base:
                logger.warning("Knowledge base is empty")
                return []
            
            # 초경량 키워드 매칭
            scored_docs = []
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for doc in self.knowledge_base:
                content = doc.get("content", "").lower()
                metadata = doc.get("metadata", {})
                category = metadata.get("category", "").lower()
                
                # 간단한 점수 계산
                score = 0
                
                # 카테고리 직접 매칭 (높은 점수)
                if any(word in category for word in query_words):
                    score += 5
                
                # 내용 키워드 매칭
                for word in query_words:
                    if word in content:
                        score += 1
                
                # 최소 점수 이상만 반환
                if score >= 1:
                    scored_docs.append({
                        "content": doc.get("content", ""),
                        "metadata": metadata,
                        "score": score
                    })
            
            # 점수순 정렬 후 상위 결과 반환
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            return scored_docs[:n_results]

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """초경량 응답 생성"""
        try:
            # OpenAI 클라이언트가 없으면 fallback 응답 사용
            if not self.openai_client:
                return self._get_fallback_response(query)

            # 컨텍스트 문서들을 간단히 결합 (최대 3개만 사용)
            context_docs = context_docs[:3]  # 메모리 절약
            context = "\n\n".join([doc["content"][:500] for doc in context_docs])  # 길이 제한

            # 간단한 프롬프트
            system_prompt = "남양주남부경찰서 AI 어시스턴트입니다. 제공된 정보를 바탕으로 도움이 되는 답변을 해주세요."

            # OpenAI API 호출 (최소 설정)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"질문: {query}\n\n참고정보: {context}"}
                ],
                max_tokens=500,  # 토큰 수 제한
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._get_fallback_response(query)

    def query(self, question: str) -> str:
        """초경량 RAG 질의응답"""
        try:
            # 간단한 캐시 확인
            cache_key = question.lower().strip()
            with self._cache_lock:
                if cache_key in self._response_cache:
                    response, _ = self._response_cache[cache_key]
                    return response
            
            # OpenAI API 키가 없으면 기본 응답
            if not self.openai_api_key or self.openai_api_key == "test-key":
                response = self._get_fallback_response(question)
            else:
                # 최소한의 문서 검색 (최대 2개)
                similar_docs = self.search_similar_documents(question, n_results=2)
                
                if not similar_docs:
                    response = "죄송합니다. 관련 정보를 찾을 수 없습니다. 다른 질문을 해주시거나 경찰서에 직접 문의해주세요."
                else:
                    # 간단한 응답 생성
                    response = self.generate_response(question, similar_docs)
            
            # 간단한 캐시 저장 (최대 10개만 유지)
            with self._cache_lock:
                if len(self._response_cache) >= 10:
                    # 가장 오래된 항목 제거
                    oldest_key = min(self._response_cache.keys(), 
                                   key=lambda k: self._response_cache[k][1])
                    del self._response_cache[oldest_key]
                self._response_cache[cache_key] = (response, time.time())
            
            return response

        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return self._get_fallback_response(question)

    def _get_fallback_response(self, question: str) -> str:
        """OpenAI API가 없을 때의 대체 응답"""
        question_lower = question.lower()

        if "가정폭력" in question_lower or "가정폭력" in question:
            return """😔 정말 힘든 상황이시군요. 가정폭력은 절대 혼자 감당해야 할 일이 아닙니다.

🛡️ **지금 가장 중요한 것은 안전입니다**

혹시 지금 당장 위험한 상황이신가요? 그렇다면 즉시 112로 신고해주세요. 경찰관이 바로 현장에 도착해서 도와드릴 거예요.

---

## 🚨 긴급신고 방법

**📞 112 신고** - 경찰서 즉시 출동  
**📞 1366 신고** - 여성긴급전화 (24시간 운영)

---

## 👮‍♀️ 신고 후 절차

1. **피해자 보호** - 안전한 곳으로 이동
2. **가해자 조사** - 상세한 조사 진행  
3. **긴급조치** - 접근금지, 퇴거명령 등

---

## 🤝 피해자 지원 서비스

• 🏠 **피해자 보호시설** 연계
• 💬 **상담서비스** 제공
• ⚖️ **법률상담**과 의료지원
• 💰 **경제적 지원** 가능

---

## 📋 중요사항

**🔍 증거보전**
- 사진, 녹음, 의료기록 등 보관
- 목격자 연락처 수집

**🔒 안전확보**
- 가해자와의 접촉 차단

---

지금 상황이 어떠신지, 더 구체적으로 말씀해주시면 더 정확한 도움을 드릴 수 있어요. 혼자 견디지 마시고 언제든 연락주세요.

**📞 남양주남부경찰서: 031-123-4567**  
*24시간 신고접수 및 대응 가능*"""

        elif "교통사고" in question_lower or "교통사고" in question:
            return """😰 아, 교통사고가 나셨군요. 많이 놀라셨을 것 같아요.

**🚑 먼저 다치신 분은 없으신가요?**  
부상자가 있으시면 즉시 119에 신고해주세요. 사람이 가장 중요하거든요.

---

## 🚨 즉시 신고

**📞 112 신고** - 경찰서 신고  
**📞 119 신고** - 부상자 응급의료

---

## ⚠️ 현장 보존

**🚗 차량 이동 금지**  
사고가 나면 차량을 움직이지 마시고 현장을 그대로 보존해주세요. 그래야 사고 원인을 정확히 파악할 수 있어요.

---

## 👮‍♂️ 경찰관 도착 시

1. **📝 사고 경위** - 자세히 설명해주세요
2. **📄 서류 준비** - 운전면허증, 차량등록증, 보험증서
3. **🏢 보험사 연락** - 보험처리 진행

---

## ⏰ 처리 시간

**⏱️ 현장조사: 1-2시간 소요**

---

혹시 사고 상황이 어떻게 되었는지, 다치신 분은 없는지 더 자세히 말씀해주시면 더 구체적인 도움을 드릴 수 있어요.

**📞 남양주남부경찰서: 031-123-4567**"""

        elif "분실물" in question_lower or "분실물" in question:
            return """😔 아, 물건을 잃어버리셨군요. 정말 속상하실 것 같아요.

**🔍 무엇을 잃어버리셨는지, 어디서 잃어버리셨는지 기억나시나요?**

---

## 📝 분실물 신고 방법

**🏢 경찰서 방문신고**  
- 직접 방문하여 신고

**💻 온라인 신고**  
- 웹사이트: www.lost112.go.kr

---

## 📋 신고 시 준비사항

**🆔 신분증** - 꼭 가져오세요  
**📝 상세 설명** - 색깔, 크기, 특징 등 자세히

---

## 📦 보관 정책

**📅 일반물품** - 3개월간 보관  
**💎 귀중품** - 1년간 보관  
**💰 수수료** - 무료

---

혹시 잃어버린 물건이 무엇인지, 어디서 잃어버리셨는지 더 자세히 말씀해주시면 찾는데 도움이 될 것 같아요.

**📞 남양주남부경찰서: 031-123-4567**"""

        else:
            return """👋 안녕하세요! 남양주남부경찰서 AI 챗봇입니다.

다음과 같은 업무를 도와드릴 수 있습니다:

---

## 🚨 신고접수

**📞 112** - 범죄신고  
**📞 110** - 교통사고 신고  
**📞 119** - 응급상황 신고

---

## 📋 민원서비스

• 🔍 **분실물** 신고 및 찾기
• 📄 **증명서** 발급
• 💬 **생활안전** 상담

---

## 🏢 경찰서 정보

**📍 주소**  
경기도 남양주시 화도읍 경춘로 1234

**📞 전화**  
031-123-4567

**🗺️ 관할구역**  
화도읍, 진접읍, 오남읍, 별내면, 수동면, 조안면

---

더 구체적인 질문을 해주시면 더 정확한 답변을 드릴 수 있습니다! 😊"""


# 전역 RAG 시스템 인스턴스
rag_system = None


def get_rag_system():
    """RAG 시스템 인스턴스 반환 (싱글톤 패턴)"""
    global rag_system
    if rag_system is None:
        try:
            rag_system = RAGSystem()
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            # RAG 시스템 초기화 실패 시 None 반환
            rag_system = None
    return rag_system

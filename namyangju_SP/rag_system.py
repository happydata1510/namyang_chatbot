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

# Vercel 환경에서는 무거운 패키지들을 조건부로 import
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.chroma_persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        
        # 캐시 초기화
        self._response_cache = {}
        self._cache_lock = threading.Lock()
        self._last_cache_cleanup = time.time()

        # ChromaDB 클라이언트 초기화 (Vercel 환경에서는 비활성화)
        self.chroma_client = None
        self.collection = None
        
        # ChromaDB가 사용 가능하고 Vercel 환경이 아닌 경우에만 초기화
        if CHROMADB_AVAILABLE and not os.getenv('VERCEL'):
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=self.chroma_persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
                
                # 컬렉션 이름
                self.collection_name = "police_knowledge"
                
                # 컬렉션 가져오기 또는 생성
                try:
                    self.collection = self.chroma_client.get_collection(self.collection_name)
                except:
                    self.collection = self.chroma_client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "경찰청 관련 지식 베이스"},
                    )
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {str(e)}")
                self.chroma_client = None
                self.collection = None

        # 임베딩 모델 초기화 (지연 로딩)
        self._embedding_model = None
        self._model_lock = threading.Lock()

        # OpenAI 모델 초기화 (API 키가 있고 LangChain이 사용 가능할 때만)
        if (self.openai_api_key and self.openai_api_key != "test-key" and 
            LANGCHAIN_AVAILABLE):
            try:
                self.llm = ChatOpenAI(
                    openai_api_key=self.openai_api_key,
                    model_name="gpt-4o-mini",
                    temperature=0.7,
                    max_tokens=1000,
                )
            except Exception as e:
                logger.warning(f"OpenAI model initialization failed: {str(e)}")
                self.llm = None
        else:
            self.llm = None
    
    @property
    def embedding_model(self):
        """지연 로딩으로 임베딩 모델 초기화"""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            with self._model_lock:
                if self._embedding_model is None:
                    try:
                        self._embedding_model = SentenceTransformer(self.embedding_model_name)
                    except Exception as e:
                        logger.warning(f"Embedding model initialization failed: {str(e)}")
                        return None
        return self._embedding_model
    
    def _cleanup_cache(self):
        """캐시 정리 (10분마다 실행)"""
        current_time = time.time()
        if current_time - self._last_cache_cleanup > 600:  # 10분
            with self._cache_lock:
                # 1시간 이상 된 캐시 항목 제거
                expired_keys = [
                    key for key, (_, timestamp) in self._response_cache.items()
                    if current_time - timestamp > 3600
                ]
                for key in expired_keys:
                    del self._response_cache[key]
                self._last_cache_cleanup = current_time

    def add_documents(self, documents: List[Dict[str, Any]]):
        """문서를 벡터 데이터베이스에 추가"""
        try:
            # ChromaDB가 초기화되지 않은 경우 스킵
            if not self.collection:
                logger.warning("ChromaDB not available, skipping document addition")
                return False
                
            texts = [doc["content"] for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]

            # 임베딩 생성
            embeddings = self.embedding_model.encode(texts).tolist()

            # ChromaDB에 추가
            self.collection.add(
                embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
            )

            logger.info(f"Added {len(documents)} documents to knowledge base")
            return True

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

    def search_similar_documents(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """유사한 문서 검색"""
        try:
            # ChromaDB가 초기화되지 않은 경우 빈 결과 반환
            if not self.collection:
                logger.warning("ChromaDB not available, returning empty search results")
                return []
                
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query]).tolist()[0]

            # 유사 문서 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            # 결과 포맷팅
            similar_docs = []
            for i in range(len(results["documents"][0])):
                similar_docs.append(
                    {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

            return similar_docs

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """컨텍스트를 기반으로 응답 생성"""
        try:
            # LLM이 없으면 fallback 응답 사용
            if not self.llm:
                return self._get_fallback_response(query)

            # 컨텍스트 문서들을 하나의 문자열로 결합
            context = "\n\n".join([doc["content"] for doc in context_docs])

            # 프롬프트 템플릿 생성
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="""당신은 남양주남부경찰서의 AI 어시스턴트입니다. 
                제공된 컨텍스트 정보를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.
                
                답변 시 다음 사항을 지켜주세요:
                1. 정확하고 신뢰할 수 있는 정보만 제공
                2. 경찰서 관련 업무에 도움이 되는 구체적인 안내
                3. 친근하고 이해하기 쉬운 언어 사용
                4. 필요시 관련 기관 연락처나 절차 안내
                5. 모르는 내용은 솔직히 말하고 적절한 기관을 안내
                
                컨텍스트 정보:
                {context}"""
                    ),
                    HumanMessage(content="{query}"),
                ]
            )

            # 프롬프트 포맷팅
            formatted_prompt = prompt_template.format_messages(
                context=context, query=query
            )

            # LLM을 통한 응답 생성
            response = self.llm.invoke(formatted_prompt)

            return response.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._get_fallback_response(query)

    def query(self, question: str) -> str:
        """RAG 시스템을 통한 질의응답"""
        try:
            # 캐시 정리
            self._cleanup_cache()
            
            # 캐시에서 응답 확인
            cache_key = question.lower().strip()
            with self._cache_lock:
                if cache_key in self._response_cache:
                    response, _ = self._response_cache[cache_key]
                    logger.info(f"Cache hit for query: {question}")
                    return response
            
            # OpenAI API 키가 없거나 테스트 키인 경우 기본 응답
            if not self.openai_api_key or self.openai_api_key == "test-key":
                response = self._get_fallback_response(question)
            else:
                # 1. 유사한 문서 검색
                similar_docs = self.search_similar_documents(question, n_results=3)
                logger.info(
                    f"Found {len(similar_docs)} similar documents for query: {question}"
                )

                if not similar_docs:
                    response = "죄송합니다. 관련 정보를 찾을 수 없습니다. 다른 질문을 해주시거나 경찰서에 직접 문의해주세요."
                else:
                    # 2. 컨텍스트를 기반으로 응답 생성
                    response = self.generate_response(question, similar_docs)
            
            # 응답을 캐시에 저장
            with self._cache_lock:
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
        rag_system = RAGSystem()
    return rag_system

"""
RAG (Retrieval-Augmented Generation) 시스템 구현
"""

import os
import json
from typing import List, Dict, Any, Set
from django.conf import settings
import logging
import threading
import time
from functools import lru_cache
import re

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

        # 동의어 및 관련어 사전
        self.synonyms = self._init_synonyms()

        # 인덱스 구축
        self._build_index()
        
        # 세션 관리 (사용자별 대화 컨텍스트)
        self.sessions = {}

        # OpenAI 클라이언트 초기화
        if (
            self.openai_api_key
            and self.openai_api_key != "test-key"
            and OPENAI_AVAILABLE
        ):
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
        """지식베이스 데이터 로드 (통합 지식베이스 사용)"""
        try:
            from namyangju_SP.knowledge_base import get_all_knowledge_base

            self.knowledge_base = get_all_knowledge_base()
            logger.info(f"Loaded {len(self.knowledge_base)} knowledge base documents")

            # Excel 지식베이스도 로드 시도
            try:
                from namyangju_SP.knowledge_base import get_excel_knowledge_base

                excel_kb = get_excel_knowledge_base()
                if excel_kb:
                    logger.info(f"Loaded {len(excel_kb)} Excel Q&A documents")
            except Exception as e:
                logger.warning(f"Could not load Excel knowledge base: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            self.knowledge_base = []

    def _init_synonyms(self) -> Dict[str, List[str]]:
        """동의어 및 관련어 사전 초기화"""
        return {
            # 학대 관련
            "학대": ["폭력", "구타", "폭행", "욕설", "협박", "위협"],
            "노인학대": ["노인", "할머니", "할아버지", "어르신"],
            "가정폭력": ["가정내폭력", "배우자폭력", "부부폭력", "가정폭행"],
            "성폭력": ["성범죄", "강간", "추행", "성추행"],
            # 신고 관련
            "신고": ["신고하", "고발", "알림", "접수"],
            "피해자": ["피해당하", "괴롭힘", "시달림"],
            "도움": ["구조", "도와주", "상담", "지원"],
            # 장소 관련
            "경찰서": ["경찰", "순경", "경찰서", "112"],
            "병원": ["의원", "병원", "의료원"],
            "시설": ["요양원", "시설", "기관"],
            # 상황 관련
            "응급": ["긴급", "위급", "응급상황", "긴급상황"],
            "위험": ["위험하", "위험한", "무서운", "두려운"],
            "긴급": ["당장", "지금", "바로", "즉시"],
            # 감정 관련
            "두려운": ["무서운", "겁나는", "불안한"],
            "힘든": ["어려운", "막막한", "어렵"],
        }

    def _build_index(self):
        """인덱스 구축 (토큰화된 단어 매핑)"""
        self.index = {}  # {token: [doc_idx, ...]}
        self.doc_tokens = []  # 문서별 토큰 집합

        for idx, doc in enumerate(self.knowledge_base):
            content = doc.get("content", "").lower()
            metadata = doc.get("metadata", {})

            # 토큰 추출
            tokens = self._extract_tokens(content)
            if metadata:
                cat = metadata.get("category", "").lower()
                tokens.update(self._extract_tokens(cat))

                # 질문이 있으면 질문도 토큰화
                question = metadata.get("question", "").lower()
                if question:
                    tokens.update(self._extract_tokens(question))

            self.doc_tokens.append(tokens)

            # 인덱스에 추가
            for token in tokens:
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append(idx)

    def _extract_tokens(self, text: str) -> Set[str]:
        """텍스트에서 토큰 추출"""
        if not text:
            return set()

        # 한글, 영문, 숫자만 허용
        text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)

        tokens = set()

        # 단어 토큰 (2글자 이상)
        words = text.split()
        for word in words:
            if len(word) >= 2:
                tokens.add(word)
                # 긴 단어의 경우 n-gram도 추가
                if len(word) >= 4:
                    for i in range(len(word) - 1):
                        if i + 2 <= len(word):
                            tokens.add(word[i : i + 2])

        return tokens

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
        """개선된 토큰 기반 문서 검색"""
        try:
            if not self.knowledge_base:
                logger.warning("Knowledge base is empty")
                return []

            # 질의 토큰 추출
            query_tokens = self._extract_tokens(query.lower())

            # 동의어 확장
            expanded_tokens = self._expand_with_synonyms(query_tokens)

            # 토큰 기반 검색
            doc_scores = {}

            # 직접 매칭 (높은 점수)
            for token in query_tokens:
                if token in self.index:
                    for doc_idx in self.index[token]:
                        if doc_idx not in doc_scores:
                            doc_scores[doc_idx] = 0
                        doc_scores[doc_idx] += 10  # 직접 매칭 높은 점수

            # 동의어 매칭 (중간 점수)
            for token in expanded_tokens:
                if token in self.index:
                    for doc_idx in self.index[token]:
                        if doc_idx not in doc_scores:
                            doc_scores[doc_idx] = 0
                        doc_scores[doc_idx] += 5  # 동의어 매칭 중간 점수

            # 카테고리 및 메타데이터 보너스
            for doc_idx in doc_scores:
                doc = self.knowledge_base[doc_idx]
                content = doc.get("content", "").lower()
                metadata = doc.get("metadata", {})
                category = metadata.get("category", "").lower()
                question = metadata.get("question", "").lower()

                # 카테고리 매칭 보너스
                for token in query_tokens:
                    if token in category:
                        doc_scores[doc_idx] += 8

                # 질문 매칭 보너스
                if question:
                    for token in query_tokens:
                        if token in question:
                            doc_scores[doc_idx] += 10

                # TF 가중치 (토큰 빈도)
                for token in query_tokens:
                    count = content.count(token)
                    if count > 0:
                        doc_scores[doc_idx] += min(count, 5)  # 최대 5점

            # 점수별 정렬
            scored_docs = []
            for doc_idx, score in doc_scores.items():
                doc = self.knowledge_base[doc_idx]
                scored_docs.append(
                    {
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "score": score,
                    }
                )

            # 점수순 정렬 후 상위 결과 반환
            scored_docs.sort(key=lambda x: x["score"], reverse=True)

            # 최소 점수 이상만 반환 (개선: 임계값 낮춤)
            results = [doc for doc in scored_docs if doc["score"] >= 5]

            return results[:n_results]

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def _expand_with_synonyms(self, tokens: Set[str]) -> Set[str]:
        """동의어로 토큰 확장"""
        expanded = set(tokens)

        for token in tokens:
            # 동의어 사전에서 찾기
            for key, synonyms in self.synonyms.items():
                if token in key or key in token:
                    expanded.update(synonyms)

            # 역방향 검색 (synonym list에서 token 찾기)
            for key, synonyms in self.synonyms.items():
                for synonym in synonyms:
                    if token in synonym or synonym in token:
                        expanded.add(key)
                        expanded.update([s for s in synonyms if s != synonym])

        return expanded

    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """초경량 응답 생성"""
        try:
            # OpenAI 클라이언트가 없으면 fallback 응답 사용
            if not self.openai_client:
                return self._get_fallback_response(query)

            # 컨텍스트 문서들을 간단히 결합 (최대 3개만 사용)
            context_docs = context_docs[:3]  # 메모리 절약
            context = "\n\n".join(
                [doc["content"][:500] for doc in context_docs]
            )  # 길이 제한

            # 간단한 프롬프트
            system_prompt = "남양주남부경찰서 AI 어시스턴트입니다. 제공된 정보를 바탕으로 도움이 되는 답변을 해주세요."

            # OpenAI API 호출 (최소 설정)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"질문: {query}\n\n참고정보: {context}",
                    },
                ],
                max_tokens=500,  # 토큰 수 제한
                temperature=0.7,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._get_fallback_response(query)

    def query(self, question: str, session_id: str = "default") -> str:
        """3단계 상담 기반 RAG 질의응답"""
        try:
            # 세션 관리
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "phase": 1,  # 1: 문제 파악, 2: 상담, 3: 해결책
                    "problem_type": None,
                    "history": []
                }
            
            session = self.sessions[session_id]
            
            # 먼저 지식베이스에서 관련 문서 검색
            similar_docs = self.search_similar_documents(question, n_results=2)
            
            # Phase별 응답 생성
            if session["phase"] == 1:
                response = self._generate_phase1_response(question, similar_docs, session)
            elif session["phase"] == 2:
                response = self._generate_phase2_response(question, similar_docs, session)
            else:  # phase == 3
                response = self._generate_phase3_response(question, similar_docs, session)
            
            # 대화 히스토리 저장
            session["history"].append({
                "question": question,
                "response": response,
                "timestamp": time.time()
            })
            
            # 최근 20개만 유지
            if len(session["history"]) > 20:
                session["history"] = session["history"][-20:]
            
            return response

        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return "죄송합니다. 문제가 발생했습니다. 다시 말씀해주시겠어요?"
    
    def _generate_phase1_response(self, question: str, similar_docs: List[Dict], session: Dict) -> str:
        """Phase 1: 문제 파악 및 기본 전화번호 제시"""
        question_lower = question.lower()
        
        # 문제 유형 파악
        problem_type = self._identify_problem_type(question)
        session["problem_type"] = problem_type
        
        # 기본 전화번호 제시
        response = self._get_emergency_numbers(problem_type)
        
        # 다음 Phase로 이동 신호
        session["phase"] = 2
        
        return response
    
    def _generate_phase2_response(self, question: str, similar_docs: List[Dict], session: Dict) -> str:
        """Phase 2: 상담사처럼 상담 진행"""
        # 사용자가 더 자세한 정보를 원하는지, 해결책으로 넘어갈 준비가 되었는지 확인
        if self._is_ready_for_solution(question):
            session["phase"] = 3
            return self._generate_phase3_response(question, similar_docs, session)
        
        # 상담 계속
        return self._generate_counseling_response(question, similar_docs, session)
    
    def _generate_phase3_response(self, question: str, similar_docs: List[Dict], session: Dict) -> str:
        """Phase 3: 해결 방법 제시"""
        problem_type = session.get("problem_type", "일반")
        
        # 지식베이스에서 해결 방법 찾기
        if similar_docs:
            solution = similar_docs[0]["content"]
            
            response = f"""해결 방법을 안내해드리겠습니다.

{'='*60}
{solution}
{'='*60}

추가로 도움이 필요하시면 언제든지 말씀해주세요.
📞 필요시 다시 연락: 남양주남부경찰서 031-123-4567"""
            return response
        
        # 기본 해결 방법
        return self._get_general_solution(problem_type)
    
    def _identify_problem_type(self, question: str) -> str:
        """질문에서 문제 유형 파악"""
        question_lower = question.lower()
        
        if "가정폭력" in question_lower or "배우자" in question_lower or "남편" in question_lower or "아내" in question_lower:
            return "가정폭력"
        elif "노인" in question_lower or "할머니" in question_lower or "할아버지" in question_lower:
            return "노인학대"
        elif "성" in question_lower or "강간" in question_lower:
            return "성폭력"
        elif "스토킹" in question_lower:
            return "스토킹"
        elif "사기" in question_lower or "보이스피싱" in question_lower:
            return "사기"
        elif "응급" in question_lower or "재난" in question_lower:
            return "응급상황"
        else:
            return "일반"
    
    def _get_emergency_numbers(self, problem_type: str) -> str:
        """문제 유형별 긴급전화번호"""
        numbers = {
            "가정폭력": {
                "title": "가정폭력 긴급 신고",
                "numbers": "📞 112 (경찰) / 📞 1366 (여성긴급전화)",
                "message": "가정폭력은 법적 보호가 가능합니다. 지금 즉시 신고하세요."
            },
            "노인학대": {
                "title": "노인학대 긴급 신고",
                "numbers": "📞 1577-1389 (노인보호전문기관) / 📞 112 (경찰)",
                "message": "노인학대 신고는 24시간 가능하며 신고자 보호가 됩니다."
            },
            "성폭력": {
                "title": "성폭력 긴급 신고",
                "numbers": "📞 112 (경찰) / 📞 117 (국번없이) (1366번 안내)",
                "message": "성폭력은 즉시 신고하세요. 의료 증거 보전이 중요합니다."
            },
            "스토킹": {
                "title": "스토킹 신고",
                "numbers": "📞 112 (경찰)",
                "message": "스토킹은 범죄입니다. 증거를 모으고 즉시 신고하세요."
            },
            "사기": {
                "title": "사기/보이스피싱 신고",
                "numbers": "📞 112 (경찰) / 📞 1332 (금융감독원)",
                "message": "계좌가 보안상 차단되어 있다면 즉시 은행에 연락하세요."
            },
            "응급상황": {
                "title": "응급상황 신고",
                "numbers": "📞 112 (경찰) / 📞 119 (소방)",
                "message": "생명이 위급한 상황은 즉시 112 또는 119로 신고하세요."
            },
        }
        
        info = numbers.get(problem_type, {
            "title": "긴급 신고",
            "numbers": "📞 112 (경찰)",
            "message": "도움이 필요하시면 언제든지 신고하세요."
        })
        
        return f"""안녕하세요. 정말 걱정되는 상황이시군요.

🚨 {info['title']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{info['numbers']}

💡 {info['message']}

📋 다음 단계:
현재 상황을 더 자세히 말씀해주시면 구체적인 도움을 드릴 수 있습니다.
- 언제부터 이런 일이 있었나요?
- 지금 안전하신가요?
- 추가로 도움을 받고 싶은 부분이 있으신가요?

계속 이야기해주세요. 지금 여기서 들어드리겠습니다."""

    def _is_ready_for_solution(self, question: str) -> bool:
        """해결책으로 넘어갈 준비가 되었는지 확인"""
        question_lower = question.lower()
        
        solution_keywords = [
            "어떻게 해야", "방법", "어떻게 하면", "해결", "대처",
            "무엇을 해야", "조치", "절차", "가이드"
        ]
        
        for keyword in solution_keywords:
            if keyword in question_lower:
                return True
        
        return False
    
    def _generate_counseling_response(self, question: str, similar_docs: List[Dict], session: Dict) -> str:
        """상담 응답 생성"""
        response_parts = []
        
        # 공감 표현
        response_parts.append("정말 힘드신 일이었을 거예요. 너무 고생이 많으셨어요.")
        
        # OpenAI가 있으면 GPT로 응답 생성
        if self.openai_client and similar_docs:
            try:
                context = "\n\n".join([doc["content"][:300] for doc in similar_docs[:2]])
                prompt = f"""상담사 역할을 해주세요. 공감하고 위로하며, 다음 정보를 바탕으로 도움이 되는 답변을 해주세요.

참고 정보:
{context}

사용자 질문: {question}

짧고 따뜻하게 답변해주세요. (100자 이내)"""
                
                result = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.7
                )
                response_parts.append(result.choices[0].message.content)
            except:
                pass
        
        # 추가 안내
        response_parts.append("\n더 구체적인 상황을 말씀해주시면 정확한 도움을 드릴 수 있어요.")
        
        return "\n\n".join(response_parts)
    
    def _get_general_solution(self, problem_type: str) -> str:
        """일반적인 해결 방법"""
        solutions = {
            "가정폭력": "가정폭력 피해자는 피해자보호시설, 상담서비스, 법률지원을 받을 수 있습니다. 112로 신고하거나 1366으로 상담받으세요.",
            "노인학대": "노인보호전문기관(1577-1389)에 신고하세요. 현장조사, 긴급조치, 법적 지원을 받을 수 있습니다.",
            "성폭력": "성폭력 피해자는 즉시 112로 신고하고 의료기관에서 진단을 받으세요. 증거보전이 중요합니다.",
            "일반": "도움이 필요하신 분은 112로 신고하거나 남양주남부경찰서(031-123-4567)로 연락하세요."
        }
        
        return solutions.get(problem_type, solutions["일반"])

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

        elif (
            "응급" in question_lower
            or "재난" in question_lower
            or "화재" in question_lower
        ):
            return """🚨 응급상황 신고 및 대응 안내

## 📞 즉시 신고

**📞 112** - 경찰 신고 및 출동 요청  
**📞 119** - 화재, 응급의료, 구조 요청

---

## ⚡ 주요 상황별 대응

### 🔥 화재
- 즉시 119 신고
- 대피 우선, 소화기 사용
- 연기 주의 (낮은 자세로 이동)

### 🏗️ 붕괴사고
- 112, 119 신고
- 안전한 장소로 대피
- 출입 통제 협조

### 🌊 자연재해
- 112 신고
- 고지대로 대피
- 전기, 가스 차단

---

## 🆘 대응절차

1. **신고접수** → 112 또는 119
2. **현장출동** → 출동 시간: 5-10분
3. **상황파악** → 신고자와 상세 내용 확인
4. **대응조치** → 즉시 조치 및 피해자 구호

---

## 🏥 피해자 구호

- 응급처치 제공
- 대피지도 및 안전 확보
- 피해자 보호 및 안심

---

## 🚦 현장관리

- 교통통제
- 출입통제
- 안전 확보

---

지금 구체적으로 어떤 상황인지 알려주시면 더 정확한 도움을 드릴 수 있습니다.

**📞 남양주남부경찰서: 031-123-4567**  
*24시간 대응체계 운영*"""

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

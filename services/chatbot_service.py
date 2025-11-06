import re
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import os
import torch
from datetime import datetime

#0825 수정
from services.genie_qwen import genie_analyze_intent, qwen_prompt_command, _ensure_utf8

# Nomic API를 사용할지 ONNX를 사용할지 설정
USE_ONNX = True  # True: ONNX 모델 사용, False: Nomic API 사용
ONNX_MODEL_PATH = "C:/Users/csw21/Downloads/nomic_embed_text.onnx/model.onnx/model.onnx"

# API fallback용
try:
    from nomic import embed
    NOMIC_API_AVAILABLE = True
except ImportError:
    NOMIC_API_AVAILABLE = False

class ChatbotService:
    def __init__(self, config, ai_models, email_service):
        self.config = config
        self.ai_models = ai_models
        self.email_service = email_service
        
        # ONNX 모델 초기화
        self.onnx_session = None
        self.tokenizer = None
        if USE_ONNX and os.path.exists(ONNX_MODEL_PATH):
            try:
                print("[🚀 ONNX] Nomic 임베딩 모델 로딩 중...")
                self.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                print("[✅ ONNX] 모델 로딩 완료!")
            except Exception as e:
                print(f"[❌ ONNX] 모델 로딩 실패: {e}")
                print("[⚠️ ONNX] Nomic API로 폴백합니다.")
        
        # 챗봇 의도 분류용 라벨 (한국어)
        self.candidate_labels = [
            "한국어 문법과 맞춤법 오류를 교정하고 수정해주세요",
            "키워드나 제목으로 이메일을 검색하고 찾아주세요",
            "김철수, 박영희 같은 특정 사람이 보낸 이메일을 찾아주세요",
            "어제, 오늘, 지난주 등 날짜로 이메일을 검색해주세요", 
            "최신 메일만, 오래된 메일만 등으로 이메일 목록을 필터링해주세요",
            "받은메일함 또는 보낸메일함에서만 이메일을 검색해주세요",
            "오늘 메일 개수, 총 메일 통계 등을 보여주세요",
            "여러 조건을 조합해서 복합적으로 이메일을 검색해주세요",
            "폰트 크기, 테마 모드, 발신자 이름, 페이지당 표시 개수, Gmail 개수 등 앱 설정을 변경해주세요"
        ]
        
        # 한국어 패턴 매칭
        self.korean_patterns = {
            "grammar": {
                "keywords": ["교정", "맞춤법", "문법", "틀렸", "고쳐", "수정"],
                "action": "grammar_correction"
            },
            "person_search": {
                "keywords": ["님", "씨"],
                "required": ["메일", "이메일"],
                "action": "person_search"
            },
            "general_search": {
                "keywords": ["찾아", "검색", "찾기"],
                "action": "email_search"
            },
            "email_stats": {
                "keywords": ["몇 개", "개수", "통계", "얼마나", "몇", "총", "합계"],
                "action": "email_statistics"
            },
            "date_search": {
                "keywords": ["어제", "오늘", "지난주", "이번주", "지난달", "이번달", "그제"],
                "action": "email_search",
                "detailed_intent": "search emails by date and time period"
            },
            "limit_search": {
                "keywords": ["개만", "최신", "최근"],
                "action": "email_search",
                "detailed_intent": "search emails with quantity limits"
            },
            "type_search": {
                "keywords": ["받은메일", "보낸메일"],
                "action": "email_search", 
                "detailed_intent": "search emails by type sent or received"
            },
            "settings_control": {
                "keywords": ["설정", "변경", "바꿔", "바꾸", "조절", "수정", "설정해", "적용", "바꿔줘", "바꿔주세요", "으로", "크기를", "폰트"],
                "action": "settings_control",
                "detailed_intent": "change application settings"
            }
        }
    
    def process_user_input(self, user_input, user_email, app_password):
        """사용자 입력 처리 (학습형)"""
        try:
            import time
            start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"[🤖 챗봇 요청 시작] 사용자: {user_email}")
            print(f"[📝 입력 명령어] '{user_input}'")
            print(f"{'='*60}")
            
            if not user_input:
                return {"error": "입력이 비어있습니다."}, 400
            
            # 🎯 우선 처리 1: 설정 입력 대기 중인지 확인
            user_input_stripped = user_input.strip()
            
            # 설정 입력 대기 상태 확인
            try:
                import os
                awaiting_name_file = os.path.join("user_sessions", f"{user_email}_awaiting_name.txt")
                awaiting_font_file = os.path.join("user_sessions", f"{user_email}_awaiting_font.txt")
                awaiting_theme_file = os.path.join("user_sessions", f"{user_email}_awaiting_theme.txt")
                
                # 발신자 이름 대기 중
                if os.path.exists(awaiting_name_file):
                    os.remove(awaiting_name_file)  # 상태 파일 삭제
                    print(f"[📧 발신자 이름 입력 완료] '{user_input_stripped}'")
                    
                    # 발신자 이름 설정 API 호출
                    import requests
                    response = requests.put(
                        f'http://localhost:5001/api/settings/GENERAL/WRITE/senderName',
                        json={
                            'email': user_email,
                            'value': user_input_stripped
                        }
                    )
                    
                    if response.status_code == 200:
                        print(f"[✅ 설정 완료] 발신자 이름 '{user_input_stripped}'로 설정됨")
                        result_msg = f"✅ 발신자 이름이 '{user_input_stripped}'(으)로 설정되었습니다! 👤"
                    else:
                        print(f"[❌ 설정 실패] API 응답: {response.status_code}")
                        result_msg = f"❌ 발신자 이름 설정에 실패했습니다."
                    
                    processing_time = time.time() - start_time
                    return {
                        "response": result_msg,
                        "action": "settings_control",
                        "confidence": 0.95,
                        "processing_time": processing_time
                    }, 200
                
                
                # 폰트 입력 대기 중
                elif os.path.exists(awaiting_font_file):
                    os.remove(awaiting_font_file)  # 상태 파일 삭제
                    print(f"[🎨 폰트 입력 완료] '{user_input_stripped}'")
                    
                    # 폰트 설정 업데이트
                    from services.settings_service import SettingsService
                    settings_service = SettingsService()
                    result = settings_service.set_setting_value(
                        user_email=user_email,
                        category='GENERAL',
                        subcategory='WRITE',
                        key='fontFamily',
                        value=user_input_stripped
                    )
                    
                    if result['success']:
                        print(f"[✅ 폰트 설정 완료] '{user_input_stripped}'로 설정됨")
                        result_msg = f"✅ 폰트가 '{user_input_stripped}'(으)로 설정되었습니다! 🎨"
                    else:
                        print(f"[❌ 폰트 설정 실패] {result.get('error', '알 수 없는 오류')}")
                        result_msg = f"❌ 폰트 설정 실패: {result.get('error', '알 수 없는 오류')}"
                    
                    processing_time = time.time() - start_time
                    return {
                        "response": result_msg,
                        "action": "settings_control",
                        "confidence": 0.95,
                        "processing_time": processing_time
                    }, 200
                
                # 테마 입력 대기 중
                elif os.path.exists(awaiting_theme_file):
                    os.remove(awaiting_theme_file)  # 상태 파일 삭제
                    print(f"[🌈 테마 입력 완료] '{user_input_stripped}'")
                    
                    # 테마 값 변환
                    theme_mapping = {
                        '다크': 'dark', '다크모드': 'dark', '어둡게': 'dark', '검정': 'dark',
                        '라이트': 'light', '라이트모드': 'light', '밝게': 'light', '흰색': 'light',
                        '시스템': 'auto', '자동': 'auto', '자동설정': 'auto'
                    }
                    
                    theme_value = theme_mapping.get(user_input_stripped, user_input_stripped.lower())
                    if theme_value not in ['dark', 'light', 'auto']:
                        theme_value = 'light'  # 기본값
                    
                    # 테마 설정 업데이트
                    from services.settings_service import SettingsService
                    settings_service = SettingsService()
                    result = settings_service.set_setting_value(
                        user_email=user_email,
                        category='GENERAL',
                        subcategory='THEME',
                        key='appearance',
                        value=theme_value
                    )
                    
                    if result['success']:
                        print(f"[✅ 테마 설정 완료] '{theme_value}'로 설정됨")
                        theme_name = {'dark': '다크 모드', 'light': '라이트 모드', 'auto': '시스템 설정 따르기'}[theme_value]
                        result_msg = f"✅ 테마가 '{theme_name}'(으)로 설정되었습니다! 🌈"
                    else:
                        print(f"[❌ 테마 설정 실패] {result.get('error', '알 수 없는 오류')}")
                        result_msg = f"❌ 테마 설정 실패: {result.get('error', '알 수 없는 오류')}"
                    
                    processing_time = time.time() - start_time
                    return {
                        "response": result_msg,
                        "action": "settings_control",
                        "confidence": 0.95,
                        "processing_time": processing_time
                    }, 200
                    
            except Exception as e:
                print(f"[⚠️ 상태 확인 실패] {e}")
            
            
            
            # 🧠 1단계: 학습된 패턴에서 찾기
            print(f"[🔍 1단계] 학습된 패턴에서 매칭 검색 시작...")
            learned_result = self._try_learned_pattern(user_email, user_input, app_password)
            if learned_result:
                processing_time = time.time() - start_time
                print(f"[⚡ 학습 패턴 매칭 성공!] 처리시간: {processing_time:.3f}초 (빠름!)")
                print(f"[✅ 학습 시스템 효과] AI 처리 없이 바로 실행됨")
                print(f"{'='*60}\n")
                return {
                    **learned_result,
                    "method": "learned_pattern",
                    "processing_time": processing_time
                }, 200
            
            # 🔍 2단계: Qwen 기반 Intent 분류
            print(f"[❌ 1단계 결과] 학습된 패턴 없음 - Qwen Intent 분류로 진행")
            print(f"[🧠 2단계] Qwen 기반 의도 분류 시작...")
            
            # Qwen 기반 의도 분류 (정확함)
            intent_result = self._classify_intent_with_qwen(user_input)
            
            # Qwen 실패 시 Nomic 폴백
            if not intent_result:
                print(f"[⚠️ Qwen 실패] Nomic 폴백으로 전환")
                intent_result = self._analyze_intent(user_input)
            
            print(f"[🎯 의도 분석 결과] {intent_result['action']} (신뢰도: {intent_result['confidence']:.3f})")
            print(f"[🔧 분석 방법] {intent_result['method']}")
            
            # 기능별 실행 (세분화된 의도 처리)
            print(f"[⚙️ 기능 실행] {intent_result['action']} 핸들러 호출 중...")
            print(f"[📋 세부 의도] {intent_result.get('detailed_intent', 'general')}")
            
            if intent_result['action'] == "grammar_correction":
                response = self._handle_grammar_correction(user_input)
            elif intent_result['action'] == "email_search":
                # 세분화된 검색 의도에 따라 다른 처리
                detailed_intent = intent_result.get('detailed_intent', '')
                
                if "date and time period" in detailed_intent:
                    response = self._handle_date_search(user_input, user_email, app_password)
                elif "quantity limits" in detailed_intent:
                    response = self._handle_limit_search(user_input, user_email, app_password)
                elif "type sent or received" in detailed_intent:
                    response = self._handle_type_search(user_input, user_email, app_password)
                elif "multiple conditions" in detailed_intent:
                    response = self._handle_complex_search(user_input, user_email, app_password)
                else:
                    response = self._handle_general_search(user_input, user_email, app_password)
            elif intent_result['action'] == "person_search":
                response = self._handle_person_search(user_input, user_email, app_password)
            elif intent_result['action'] == "email_statistics":
                response = self._handle_email_statistics(user_input, user_email, app_password)
            elif intent_result['action'] == "settings_control":
                response = self._handle_settings_control(user_input, user_email, intent_result.get('details', ''))
            else:
                response = self._handle_unknown_intent()
            
            # 🔥 3단계: AI 처리 성공 시 학습 저장 (오류가 없는 경우에만)
            processing_time = time.time() - start_time
            
            # 응답이 성공적이고 오류가 없는 경우에만 저장
            if response and not isinstance(response, dict) or (isinstance(response, dict) and not response.get('error')):
                print(f"[💾 3단계] 성공적인 응답 - 학습 데이터 저장 시작...")
                
                # 저장할 응답 데이터 준비
                save_response = response
                if isinstance(response, dict) and 'results' in response:
                    # 검색 결과인 경우 간단한 요약만 저장
                    save_response = f"검색 완료: {len(response.get('results', []))}개 결과"
                
                self._auto_save_learned_command(user_email, user_input, intent_result, save_response)
            else:
                print(f"[⚠️ 3단계] 응답에 오류 있음 - 학습 데이터 저장 생략")
            
            print(f"[⏱️ 총 처리시간] {processing_time:.3f}초 (AI 처리 포함)")
            print(f"[📚 다음 실행] 동일/유사 명령어는 {processing_time:.3f}초 → 0.05초로 단축됨")
            print(f"{'='*60}\n")
            
            return {
                "response": response,
                "action": intent_result['action'],
                "confidence": float(intent_result['confidence']),
                "detected_intent": intent_result['action'],
                "detection_method": intent_result['method'],
                "method": "ai_processing",
                "processing_time": processing_time
            }, 200
            
        except Exception as e:
            print(f"[❗챗봇 오류] {str(e)}")
            return {"error": str(e)}, 500
    
    def _get_embeddings(self, texts):
        """텍스트 임베딩 생성 (ONNX 우선, API 폴백)"""
        if self.onnx_session and self.tokenizer:
            # ONNX 모델 사용
            try:
                print(f"[🚀 챗봇 ONNX] 임베딩 생성 시작 - {len(texts)}개 텍스트")
                embeddings = []
                for i, text in enumerate(texts):
                    inputs = self.tokenizer(
                        text, 
                        padding="max_length", 
                        max_length=128, 
                        truncation=True,
                        return_tensors="np"
                    )
                    
                    outputs = self.onnx_session.run(None, {
                        "input_tokens": inputs["input_ids"].astype(np.int32),
                        "attention_masks": inputs["attention_mask"].astype(np.float32)
                    })
                    embeddings.append(outputs[0][0])  # 첫 번째 출력의 첫 번째 벡터
                    print(f"[✅ 챗봇 ONNX] 텍스트 {i+1}/{len(texts)} 임베딩 완료")
                
                print(f"[🎉 챗봇 ONNX] 전체 임베딩 생성 완료!")
                return {'embeddings': embeddings}
            except Exception as e:
                print(f"[⚠️ ONNX] 임베딩 생성 실패: {e}")
                # API로 폴백
        
        # Nomic API 사용
        if NOMIC_API_AVAILABLE:
            from nomic import embed
            return embed.text(texts, model='nomic-embed-text-v1', task_type='classification')
        else:
            raise Exception("임베딩 모델을 사용할 수 없습니다.")
    
    def _qwen_analyze_intent(self, user_input):
        """Qwen 기반 정확한 의도 분석"""
        try:
            # Qwen 모델 로딩 확인
            if not self.ai_models.load_qwen_model():
                print("[⚠️ Qwen 모델 없음 - 기존 방식으로 폴백]")
                return self._analyze_intent(user_input)
            
            print(f"[🤖 Qwen 의도 분석] 입력: '{user_input}'")
            
            # 간단하고 효과적인 Qwen 프롬프트
            prompt = f"""명령어: "{user_input}"

의도 분류:
1. grammar_correction: 문법/맞춤법 교정
2. email_search: 이메일 검색
3. person_search: 특정 사람 메일 찾기
4. settings_control: 설정 변경

예시:
"폰트 18로 바꿔줘" → settings_control, font_size_18
"다크모드로" → settings_control, theme_dark
"김철수님 메일" → person_search, 김철수
"메일 검색" → email_search, general

응답 형식: action, keyword

분석:"""

            # Qwen 실행
            inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.ai_models.qwen_model.device)
            
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.1,  # 낮은 온도로 일관성 확보
                    do_sample=True,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                    pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                )
            
            generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"[🔍 디버그] 전체 생성된 텍스트 길이: {len(generated_text)}")
            print(f"[🔍 디버그] 프롬프트 길이: {len(prompt)}")
            print(f"[🔍 디버그] 전체 생성 텍스트 일부: {generated_text[:200]}...")
            
            # 프롬프트 부분 제거하고 응답만 추출
            if "분석:" in generated_text:
                qwen_response = generated_text.split("분석:")[-1].strip()
                print(f"[🔍 디버그] '분석:' 기준으로 분리")
            else:
                qwen_response = generated_text[len(prompt):].strip()
                print(f"[🔍 디버그] 프롬프트 길이 기준으로 분리")
            
            print(f"[🤖 Qwen 원본 응답] {qwen_response}")
            
            # 새로운 단순 형식 파싱: "action, keyword"
            import re
            
            # "action, keyword" 형식 파싱
            if qwen_response:
                # 첫 번째 줄만 사용 (여러 줄일 수 있음)
                first_line = qwen_response.split('\n')[0].strip()
                
                # "action, keyword" 형식으로 파싱
                if ',' in first_line:
                    parts = first_line.split(',', 1)
                    if len(parts) >= 2:
                        action = parts[0].strip()
                        keyword = parts[1].strip()
                        
                        print(f"[✅ Qwen 파싱 성공] action='{action}', keyword='{keyword}'")
                        
                        return {
                            'action': action,
                            'confidence': 0.9,
                            'method': 'qwen_ai_simple',
                            'detailed_intent': keyword,
                            'qwen_raw': qwen_response
                        }
            
            # 파싱 실패 시 키워드 기반 분석
            print(f"[🔄 Qwen 폴백] 응답 파싱 실패, 키워드 기반 분석으로 전환")
            return self._parse_qwen_response_fallback(user_input, qwen_response)
            
        except Exception as e:
            print(f"[❗ Qwen 의도 분석 오류] {str(e)}")
            # 오류 시 기존 방식으로 폴백
            return self._analyze_intent(user_input)
    
    

    def _parse_qwen_response_fallback(self, user_input, qwen_response):
        """Qwen 응답 파싱 실패 시 폴백 분석"""
        user_lower = user_input.lower()
        response_lower = qwen_response.lower()
        
        # 키워드 기반 의도 결정
        if any(word in user_lower for word in ["받은메일만", "받은메일", "받은편지함"]) and "검색" in user_lower:
            return {'action': 'type_search', 'confidence': 0.8, 'method': 'qwen_fallback', 'detailed_intent': 'received_only'}
        elif any(word in user_lower for word in ["보낸메일만", "보낸메일", "보낸편지함"]) and "검색" in user_lower:
            return {'action': 'type_search', 'confidence': 0.8, 'method': 'qwen_fallback', 'detailed_intent': 'sent_only'}
        elif "님" in user_lower or "씨" in user_lower:
            return {'action': 'person_search', 'confidence': 0.8, 'method': 'qwen_fallback', 'detailed_intent': 'person'}
        elif re.search(r'\d+개', user_lower) or "최신" in user_lower:
            return {'action': 'limit_search', 'confidence': 0.8, 'method': 'qwen_fallback', 'detailed_intent': 'limit'}
        elif any(word in user_lower for word in ["어제", "오늘", "지난주", "이번주"]):
            return {'action': 'date_search', 'confidence': 0.8, 'method': 'qwen_fallback', 'detailed_intent': 'date'}
        elif any(word in user_lower for word in ["교정", "맞춤법", "문법"]):
            return {'action': 'grammar_correction', 'confidence': 0.8, 'method': 'qwen_fallback', 'detailed_intent': 'grammar'}
        elif any(word in user_lower for word in ["개수", "통계", "몇개"]):
            return {'action': 'email_statistics', 'confidence': 0.8, 'method': 'qwen_fallback', 'detailed_intent': 'stats'}
        elif any(word in user_lower for word in ["폰트", "글꼴", "크기", "글자"]) and any(word in user_lower for word in ["바꿔", "바꿔줘", "설정", "변경", "으로"]):
            return {'action': 'settings_control', 'confidence': 0.85, 'method': 'qwen_fallback', 'detailed_intent': 'font_settings'}
        elif any(word in user_lower for word in ["다크모드", "라이트모드", "테마"]) and any(word in user_lower for word in ["바꿔", "바꿔줘", "설정", "변경"]):
            return {'action': 'settings_control', 'confidence': 0.85, 'method': 'qwen_fallback', 'detailed_intent': 'theme_settings'}
        else:
            return {'action': 'email_search', 'confidence': 0.7, 'method': 'qwen_fallback', 'detailed_intent': 'general'}

    def _qwen_analyze_intent(self, user_input):
        """Qwen을 사용한 정확한 의도 분류 (메인)"""
        try:
            if not self.ai_models.load_qwen_model():
                print("[⚠️ Qwen 모델 로딩 실패 - Nomic으로 폴백]")
                return None
            
            prompt = f"""당신은 이메일 클라이언트의 챗봇입니다. 사용자 입력의 의도를 정확히 분류하세요.

가능한 의도:
1. grammar_correction - 맞춤법/문법 교정 요청
2. email_search - 키워드로 이메일 검색
3. person_search - 특정 사람의 이메일 찾기
4. email_statistics - 이메일 통계 조회
5. settings_control - 앱 설정 변경 (폰트, 테마, 페이지, 발신자 이름 등)
6. date_search - 날짜별 이메일 검색
7. type_search - 받은메일/보낸메일 검색
8. limit_search - 개수 제한 검색

사용자 입력: "{user_input}"

의도를 한 단어로만 답하세요 (예: settings_control):"""
            
            # Qwen 실행
            inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.ai_models.qwen_model.device)
            
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                    pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                )
            
            generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            intent = response.strip().lower()
            
            # 유효한 의도인지 확인
            valid_intents = ['grammar_correction', 'email_search', 'person_search', 
                           'email_statistics', 'settings_control', 'date_search',
                           'type_search', 'limit_search']
            
            if intent in valid_intents:
                print(f"[✅ Qwen 의도 분류] {intent} (신뢰도: 높음)")
                return {'action': intent, 'confidence': 0.95, 'method': 'qwen_main'}
            else:
                print(f"[⚠️ Qwen 애매한 응답] {intent}")
                return None
                
        except Exception as e:
            print(f"[❌ Qwen 의도 분류 오류] {str(e)}")
            return None
    
    def _analyze_intent(self, user_input):
        """의도 분석 (Qwen 우선, Nomic 폴백)"""
        
        # 1. Qwen으로 의도 분석 (메인)
        qwen_result = self._qwen_analyze_intent(user_input)
        if qwen_result and qwen_result['confidence'] >= 0.9:
            return qwen_result
        
        # 2. Qwen이 애매하면 Nomic 임베딩으로 보조
        # 영어 Embedding 기반 분류
        try:
            text_inputs = [user_input] + self.candidate_labels
            result = self._get_embeddings(text_inputs)
            
            embedding_list = result['embeddings']
            email_embedding = [embedding_list[0]]
            label_embeddings = embedding_list[1:]
            
            scores = cosine_similarity(email_embedding, label_embeddings)[0]
            best_index = scores.argmax()
            embedding_score = scores[best_index]
            embedding_label = self.candidate_labels[best_index]
            
        except Exception as e:
            print(f"[⚠️ Embedding 분류 실패] {str(e)}")
            embedding_score = 0.0
            embedding_label = "unknown"
        
        # 3. 한국어 키워드 기반 분류
        korean_result = self._analyze_korean_patterns(user_input)
        
        # 3. 최종 의도 결정
        embedding_action_map = {
            "한국어 문법과 맞춤법 오류를 교정하고 수정해주세요": "grammar_correction",
            "키워드나 제목으로 이메일을 검색하고 찾아주세요": "email_search",
            "김철수, 박영희 같은 특정 사람이 보낸 이메일을 찾아주세요": "person_search",
            "어제, 오늘, 지난주 등 날짜로 이메일을 검색해주세요": "email_search", 
            "최신 메일만, 오래된 메일만 등으로 이메일 목록을 필터링해주세요": "email_search",
            "받은메일함 또는 보낸메일함에서만 이메일을 검색해주세요": "email_search",
            "오늘 메일 개수, 총 메일 통계 등을 보여주세요": "email_statistics",
            "여러 조건을 조합해서 복합적으로 이메일을 검색해주세요": "email_search",
            "폰트 크기, 테마 모드, 발신자 이름, 페이지당 표시 개수, Gmail 개수 등 앱 설정을 변경해주세요": "settings_control"
        }
        
        embedding_action = embedding_action_map.get(embedding_label, "unknown")
        embedding_threshold = 0.25
        
        # 4. 최종 의도 결정 (Qwen 결과 우선)
        # Qwen 결과가 있으면 우선 사용
        if qwen_result:
            # Nomic이나 한국어 패턴으로 보완
            if embedding_score >= 0.7 and embedding_action == qwen_result['action']:
                # Qwen과 Nomic이 일치하면 신뢰도 상승
                qwen_result['confidence'] = min(0.99, qwen_result['confidence'] + 0.1)
                print(f"[🎯 의도 일치] Qwen과 Nomic 일치 - 신뢰도 상승")
            return qwen_result
        
        # Qwen이 실패했을 때 기존 로직
        if korean_result["confidence"] >= 0.3 and korean_result["confidence"] > embedding_score:
            return {
                'action': korean_result["action"],
                'confidence': korean_result["confidence"],
                'method': 'korean_pattern',
                'detailed_intent': korean_result.get('detailed_intent', '')
            }
        elif embedding_score >= embedding_threshold:
            return {
                'action': embedding_action,
                'confidence': embedding_score,
                'method': 'nomic_fallback',
                'detailed_intent': embedding_label
            }
        else:
            # 모든 방법이 실패하면 Qwen 폴백 사용
            return self._qwen_fallback_analyze(user_input)
    
    def _analyze_korean_patterns(self, user_input):
        """한국어 패턴 분석"""
        user_input_lower = user_input.lower()
        
        korean_result = {"action": None, "confidence": 0.0, "matched_keywords": []}
        
        # 숫자 패턴 체크 (예: "5개", "10개만")
        import re
        has_number_limit = bool(re.search(r'\d+개', user_input_lower))
        
        for pattern_name, pattern_info in self.korean_patterns.items():
            matched_keywords = []
            
            # 일반 키워드 매칭
            for keyword in pattern_info["keywords"]:
                if keyword in user_input_lower:
                    matched_keywords.append(keyword)
            
            # 필수 키워드 확인 (person_search용)
            if "required" in pattern_info:
                required_found = any(req in user_input_lower for req in pattern_info["required"])
                if not required_found:
                    continue
            
            # 신뢰도 계산
            if matched_keywords:
                confidence = len(matched_keywords) / len(pattern_info["keywords"])
                
                # person_search는 특별 처리
                if pattern_name == "person_search" and "required" in pattern_info:
                    confidence += 0.3
                
                # limit_search는 숫자 패턴이 있으면 신뢰도 증가
                if pattern_name == "limit_search" and has_number_limit:
                    confidence += 0.5
                    matched_keywords.append("숫자개수")
                
                # settings_control는 특별 처리 (폰트/테마/설정 관련 키워드 조합)
                if pattern_name == "settings_control":
                    # 설정 관련 키워드 정의
                    font_keywords = ["폰트", "글꼴", "크기", "글자", "px", "포인트"]
                    theme_keywords = ["테마", "다크모드", "라이트모드", "어두운", "밝은"]
                    sender_keywords = ["이름", "발신자", "보내는", "사람", "sender"]
                    gmail_keywords = ["gmail", "메일", "개수", "가져오", "fetch"]
                    page_keywords = ["페이지", "목록", "리스트", "보여", "표시", "개씩", "씩", "한 페이지"]
                    action_keywords = ["바꿔", "바꿔줘", "바꿔주세요", "변경", "설정", "조절", "으로", "설정해", "설정해줘", "해줘"]
                    
                    has_font = any(kw in user_input_lower for kw in font_keywords)
                    has_theme = any(kw in user_input_lower for kw in theme_keywords)
                    has_sender = any(kw in user_input_lower for kw in sender_keywords)
                    has_gmail = any(kw in user_input_lower for kw in gmail_keywords)
                    has_page = any(kw in user_input_lower for kw in page_keywords)
                    has_action = any(kw in user_input_lower for kw in action_keywords)
                    
                    # 확실한 설정 변경 패턴들
                    if (has_font or has_theme or has_sender or has_gmail or has_page) and has_action:
                        confidence = 0.95  # 매우 높은 신뢰도로 설정
                        matched_keywords.extend(["확실한_설정_변경"])
                        setting_type = ""
                        if has_font: setting_type = "폰트"
                        elif has_theme: setting_type = "테마"
                        elif has_sender: setting_type = "발신자"
                        elif has_gmail: setting_type = "Gmail"
                        elif has_page: setting_type = "페이지"
                        print(f"[🎯 설정 강화] {setting_type}+액션 조합 감지 → 신뢰도 0.95")
                    elif has_action and len(matched_keywords) >= 2:
                        confidence += 0.4  # 액션 키워드 + 다수 매칭시 신뢰도 증가
                        print(f"[🎯 설정 강화] 액션+복수키워드 조합 → 신뢰도 +0.4")
                
                if confidence > korean_result["confidence"]:
                    korean_result = {
                        "action": pattern_info["action"],
                        "confidence": confidence,
                        "matched_keywords": matched_keywords,
                        "detailed_intent": pattern_info.get("detailed_intent", "")
                    }
        
        return korean_result
    
    def _handle_grammar_correction(self, user_input):
        """문법 교정 처리"""
        try:
            # Qwen으로 교정할 텍스트 정확 추출
            correction_text = self._extract_grammar_text_with_qwen(user_input)
            
            if not correction_text:
                return "📝 **문법 및 맞춤법 교정**\n\n교정하고 싶은 텍스트를 입력해주세요.\n\n예시: '안녕하세요. 제가 오늘 회의에 참석못할것 같습니다' 교정해주세요"
            
            # Qwen 로컬 모델 사용
            if self.ai_models.load_qwen_model():
                try:
                    prompt = f"""<|im_start|>system
당신은 전문 교정 편집자입니다.
<|im_end|>
<|im_start|>user
다음 텍스트의 맞춤법, 문법, 띄어쓰기를 교정해주세요.

원본 텍스트:
"{correction_text}"

교정 지침:
1. 맞춤법 오류 수정
2. 문법 오류 수정  
3. 띄어쓰기 수정
4. 자연스러운 표현으로 개선
5. 원래 의미는 유지

교정된 텍스트:
<|im_end|>
<|im_start|>assistant
"""
                    
                    inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt").to(self.ai_models.qwen_model.device)
                    
                    import torch
                    with torch.no_grad():
                        outputs = self.ai_models.qwen_model.generate(
                            **inputs,
                            max_new_tokens=200,
                            temperature=0.3,
                            do_sample=True,
                            top_p=0.9,
                            eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                            pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                        )
                    
                    generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    if "assistant" in generated_text:
                        corrected_text = generated_text.split("assistant")[-1].strip()
                    else:
                        corrected_text = generated_text[len(prompt):].strip()
                    
                    return f"""📝 **문법 및 맞춤법 교정 완료**

**원본:**
{correction_text}

**교정된 텍스트:**
{corrected_text}

✅ **AI 교정이 완료되었습니다!**"""
                    
                except Exception as e:
                    print(f"[⚠️ Qwen 문법 교정 실패] {str(e)}")
                    return self._simple_grammar_correction(correction_text)
            else:
                # Qwen 모델 로딩 실패 시 간단한 규칙 기반 교정
                return self._simple_grammar_correction(correction_text)
                
        except Exception as e:
            return "❌ 문법 교정 처리 중 오류가 발생했습니다."
    
    def _simple_grammar_correction(self, text):
        """간단한 규칙 기반 교정"""
        simple_corrections = {
            "데이타": "데이터", "컴퓨타": "컴퓨터", "셋팅": "설정",
            "미팅": "회의", "해야되는": "해야 하는", "할수있는": "할 수 있는",
            "못할것": "못할 것", "참석못할": "참석하지 못할"
        }
        
        corrected_simple = text
        applied_corrections = []
        
        for wrong, correct in simple_corrections.items():
            if wrong in corrected_simple:
                corrected_simple = corrected_simple.replace(wrong, correct)
                applied_corrections.append(f"'{wrong}' → '{correct}'")
        
        if applied_corrections:
            return f"""📝 **간단 맞춤법 교정**

**원본:** {text}
**교정된 텍스트:** {corrected_simple}

**적용된 교정:**
{chr(10).join('• ' + correction for correction in applied_corrections)}"""
        else:
            return f"📝 **교정 검토 완료**\n\n현재 텍스트에서 명백한 오류를 발견하지 못했습니다."
    
    
    def _translate_korean_to_english(self, text):
        """한국어를 영어로 번역"""
        korean_to_english = {
            "고양이": "cute cat", "강아지": "cute dog", "꽃": "beautiful flowers",
            "바다": "ocean and waves", "산": "mountains and nature", "석양": "beautiful sunset",
            "하늘": "blue sky with clouds", "숲": "forest and trees", "도시": "modern city",
            "자동차": "modern car", "집": "beautiful house", "사람": "person"
        }
        
        english_text = text
        for korean, english in korean_to_english.items():
            if korean in text:
                english_text = english_text.replace(korean, english)
        
        # 한국어가 남아있으면 기본 프롬프트 생성
        if any(ord(char) > 127 for char in english_text):
            english_text = f"a beautiful {text}"
        
        return english_text
    
    def _handle_general_search(self, user_input, user_email, app_password):
        """일반 이메일 검색 (고급 기능 포함)"""
        try:
            print(f"[🔍 고급 검색 시작] 입력: '{user_input}'")
            
            # Qwen으로 날짜, 개수 제한 파싱
            qwen_date = self._extract_date_with_qwen(user_input)
            date_filter = self._convert_date_type_to_filter(qwen_date) if qwen_date else None
            
            qwen_limit = self._extract_limit_with_qwen(user_input)
            limit_count = qwen_limit
            mail_type_filter = self._parse_mail_type_keywords(user_input)
            
            # 검색 키워드 추출 (파싱된 키워드들 제거)
            # Qwen으로 검색 키워드 정확 추출
            search_keywords = self._extract_keyword_with_qwen(user_input)
            
            if not search_keywords:
                # 폴백: 기존 방식으로 추출
                search_keywords = user_input.lower()
                remove_words = [
                    "찾아줘", "찾아주세요", "검색해줘", "검색", "find", "search", "메일", "이메일", "email",
                    "오늘", "어제", "이번주", "이번 주", "지난주", "이번달", "이번 달", "지난달",
                    "today", "yesterday", "this week", "last week", "this month", "last month",
                    "받은", "보낸", "받은메일", "보낸메일", "수신", "발신", "inbox", "sent",
                    "개만", "개까지", "최근", "최신", "처음", "상위"
                ]
                import re
                search_keywords = re.sub(r'\d+\s*개\s*(만|까지)*', '', search_keywords)
                search_keywords = re.sub(r'최근\s*\d+\s*일', '', search_keywords)
                for word in remove_words:
                    search_keywords = search_keywords.replace(word, "").strip()
                print(f"[⚠️ 폴백] 기존 방식으로 추출된 키워드: '{search_keywords}'")
            
            # 남은 키워드가 없으면 기본 안내
            if not search_keywords and not date_filter and not mail_type_filter:
                return "🔍 **메일 검색**\n\n검색하고 싶은 키워드를 입력해주세요.\n\n💡 **고급 검색 예시:**\n• '회의 관련 메일 찾아줘'\n• '어제 받은 메일 보여줘'\n• '김철수님 지난주 메일'\n• '최근 5개 메일만'\n• '받은메일만 검색'"
            
            # 키워드가 없어도 날짜/타입 필터가 있으면 검색 진행
            if not search_keywords:
                search_keywords = ""  # 빈 문자열로 모든 메일 검색
            
            print(f"[🎯 최종 검색 키워드] '{search_keywords}'")
            
            # ✅ DB에서 이메일 검색 실행 (고급 옵션 포함)
            try:
                found_emails = self._search_emails_in_db(
                    user_email, 
                    search_keywords, 
                    max_results=50,
                    date_filter=date_filter,
                    mail_type_filter=mail_type_filter,
                    limit_count=limit_count
                )
                
                if found_emails:
                    # 검색 조건 정보 생성
                    search_info = []
                    
                    if search_keywords:
                        search_info.append(f"키워드: '{search_keywords}'")
                    
                    if date_filter:
                        date_type = date_filter.get('type', 'unknown')
                        if date_type == 'today':
                            search_info.append("날짜: 오늘")
                        elif date_type == 'yesterday':
                            search_info.append("날짜: 어제")
                        elif date_type == 'this_week':
                            search_info.append("날짜: 이번주")
                        elif date_type == 'last_week':
                            search_info.append("날짜: 지난주")
                        elif date_type == 'this_month':
                            search_info.append("날짜: 이번달")
                        elif date_type == 'last_month':
                            search_info.append("날짜: 지난달")
                        elif 'recent_' in date_type and '_days' in date_type:
                            days = date_type.split('_')[1]
                            search_info.append(f"날짜: 최근 {days}일")
                    
                    if mail_type_filter:
                        type_name = "받은메일" if mail_type_filter == 'inbox' else "보낸메일"
                        search_info.append(f"타입: {type_name}")
                    
                    if limit_count:
                        search_info.append(f"개수: {limit_count}개 제한")
                    
                    search_condition = " | ".join(search_info) if search_info else "전체 검색"
                    
                    result = f"🔍 **고급 검색 결과**\n\n📋 **검색 조건**: {search_condition}\n📧 **찾은 메일**: **{len(found_emails)}개**\n\n"
                    
                    for i, mail_info in enumerate(found_emails[:5], 1):  # 최대 5개만 표시
                        result += f"**📬 {i}번째 메일**\n"
                        result += f"📋 **제목**: {mail_info['subject']}\n"
                        result += f"👤 **발신자**: {mail_info['from']}\n"
                        result += f"📅 **날짜**: {mail_info['date']}\n"
                        
                        # 요약이 있으면 표시
                        if mail_info.get('summary') and mail_info['summary'] != '요약 없음':
                            result += f"📝 **요약**: {mail_info['summary']}\n"
                        elif mail_info['preview']:
                            result += f"💬 **미리보기**: {mail_info['preview'][:100]}{'...' if len(mail_info['preview']) > 100 else ''}\n"
                        
                        # 분류가 있으면 표시
                        if mail_info.get('classification') and mail_info['classification'] != 'unknown':
                            result += f"🏷️ **분류**: {mail_info['classification']}\n"
                        
                        result += "─────────────\n"
                    
                    if len(found_emails) > 5:
                        result += f"📊 **더 있음**: 총 {len(found_emails)}개 중 상위 5개만 표시\n"
                    
                    result += "\n💡 더 정확한 검색을 위해 구체적인 키워드를 사용해보세요."
                    return result
                else:
                    return f"🔍 **검색 결과**\n\n키워드: '{search_keywords}'\n\n❌ 관련된 메일을 찾을 수 없습니다.\n\n💡 **검색 팁**:\n• 다른 키워드로 시도\n• 발신자 이름이나 이메일 주소로 검색\n• 메일 제목의 일부로 검색"
                    
            except Exception as e:
                return f"❌ 메일 검색 중 오류가 발생했습니다.\n\n오류: {str(e)}"
                
        except Exception as e:
            return "❌ 검색 처리 중 오류가 발생했습니다."
    
    def _handle_person_search(self, user_input, user_email, app_password):
        """특정 사람 메일 검색"""
        try:
            # Qwen으로 사람 이름/이메일 정확 추출
            extract_type, search_target = self._extract_person_or_email_with_qwen(user_input)
            
            if not search_target or len(search_target.strip()) < 2:
                # 간단한 추출 방법
                words = user_input.split()
                potential_targets = []
                
                for word in words:
                    if "@" in word and "." in word:  # 이메일 주소
                        potential_targets.append(word)
                    elif len(word) >= 2 and len(word) <= 4 and word.replace(" ", "").isalpha():  # 한국어 이름
                        potential_targets.append(word)
                
                if potential_targets:
                    search_target = potential_targets[0]
                else:
                    return "👤 **사람별 메일 검색**\n\n찾고 싶은 사람의 이름이나 이메일 주소를 명확히 알려주세요.\n\n예시:\n• '김철수님의 메일'\n• 'john@company.com 메일'"
            
            try:
                # Qwen으로 고급 검색 옵션 파싱
                qwen_date = self._extract_date_with_qwen(user_input)
                date_filter = self._convert_date_type_to_filter(qwen_date) if qwen_date else None
                
                qwen_limit = self._extract_limit_with_qwen(user_input)
                limit_count = qwen_limit
                mail_type_filter = self._parse_mail_type_keywords(user_input)
                
                print(f"[🔍 사람별 고급 검색] 타입: {extract_type}, 대상: '{search_target}'")
                
                # ✅ DB에서 사람별 이메일 검색 실행 (고급 옵션 포함)
                found_emails = self._search_emails_in_db(
                    user_email, 
                    search_target, 
                    max_results=100,
                    date_filter=date_filter,
                    mail_type_filter=mail_type_filter,
                    limit_count=limit_count
                )
                
                # 발신자 정보로 필터링
                person_emails = []
                search_lower = search_target.lower()
                
                for email_info in found_emails:
                    from_field = email_info['from'].lower()
                    if (search_lower in from_field or 
                        any(part.strip() in from_field for part in search_lower.split() if part.strip())):
                        person_emails.append(email_info)
                        
                        if len(person_emails) >= 10:
                            break
                
                if person_emails:
                    result = f"👤 **사람별 메일 검색 결과**\n\n🎯 검색 대상: **{search_target}**\n📧 발견된 메일: **{len(person_emails)}개**\n\n"
                    
                    for i, mail_info in enumerate(person_emails[:5], 1):  # 최대 5개만 표시
                        result += f"**📬 {i}번째 메일**\n"
                        result += f"📋 **제목**: {mail_info['subject']}\n"
                        result += f"👤 **발신자**: {mail_info['from']}\n"
                        result += f"📅 **날짜**: {mail_info['date']}\n"
                        
                        # 요약이 있으면 표시
                        if mail_info.get('summary') and mail_info['summary'] != '요약 없음':
                            result += f"📝 **요약**: {mail_info['summary']}\n"
                        elif mail_info['preview']:
                            result += f"💬 **미리보기**: {mail_info['preview'][:100]}{'...' if len(mail_info['preview']) > 100 else ''}\n"
                        
                        # 분류가 있으면 표시
                        if mail_info.get('classification') and mail_info['classification'] != 'unknown':
                            result += f"🏷️ **분류**: {mail_info['classification']}\n"
                        
                        result += "─────────────\n"
                    
                    if len(person_emails) > 5:
                        result += f"📊 **더 있음**: 총 {len(person_emails)}개 중 상위 5개만 표시\n"
                    
                    result += "\n💡 특정 메일을 자세히 보려면 메일 리스트에서 확인하세요."
                    return result
                else:
                    return f"👤 **사람별 메일 검색 결과**\n\n🎯 검색 대상: **{search_target}**\n\n❌ 해당 사람의 메일을 찾을 수 없습니다.\n\n💡 **검색 팁**:\n• 정확한 이름이나 이메일 주소로 재시도\n• 이메일 주소 전체 입력\n• 한글 이름의 경우 성함으로만 검색"
                    
            except Exception as e:
                return f"❌ 사람별 메일 검색 중 오류가 발생했습니다.\n\n오류: {str(e)}"
                
        except Exception as e:
            return "❌ 사람 검색 처리 중 오류가 발생했습니다."
    
    def _extract_person_or_email_with_qwen(self, user_input):
        """Qwen을 사용하여 사람 이름이나 이메일 주소를 정확히 추출"""
        try:
            print(f"[🤖 Qwen 추출 시작] 사람/이메일 추출: '{user_input}'")
            
            # Qwen 모델 로딩
            if not hasattr(self.ai_models, 'qwen_model') or self.ai_models.qwen_model is None:
                print("[🤖 Qwen 모델 로딩 시작]")
                self.ai_models.load_qwen_model()
            
            prompt = f"""한국어 명령에서 사람 이름이나 이메일 주소를 추출하세요.
형식: type|값

타입:
- person: 사람 이름 (김철수, 박영희, 교수님 등)
- email: 이메일 주소 (@포함)

예시:
"최수운 이메일 찾아줘" → person|최수운
"김철수님 메일 보여줘" → person|김철수
"abc@gmail.com에서 온 메일" → email|abc@gmail.com
"교수님 메일" → person|교수님
"John의 메일" → person|John
"팀장님 이메일" → person|팀장

입력: "{user_input}"
결과:"""
            
            # 토큰화
            inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt")
            
            # 생성
            import torch
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=20,  # 짧은 응답만 필요
                    do_sample=False,
                    temperature=0.1,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                    pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                )
            
            generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 응답 추출
            if "결과:" in generated_text:
                qwen_response = generated_text.split("결과:")[-1].strip()
            else:
                qwen_response = generated_text[len(prompt):].strip()
            
            print(f"[🤖 Qwen 응답] {qwen_response}")
            
            # 응답 파싱: "type|value" 형식
            lines = qwen_response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line and not line.startswith('-'):
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        extract_type = parts[0].strip()
                        extract_value = parts[1].strip()
                        
                        # 유효한 타입인지 확인
                        if extract_type in ['person', 'email']:
                            print(f"[✅ 추출 성공] {extract_type} = '{extract_value}'")
                            return extract_type, extract_value
            
            print(f"[❌ 추출 실패] 파싱할 수 없는 응답: '{qwen_response}'")
            return None, None
            
        except Exception as e:
            print(f"[❗ Qwen 추출 오류] {str(e)}")
            return None, None

    def _extract_grammar_text_with_qwen(self, user_input):
        """Qwen을 사용하여 교정할 텍스트만 정확히 추출"""
        try:
            print(f"[🤖 Qwen 교정 텍스트 추출] '{user_input}'")
            
            # Qwen 모델 로딩
            if not hasattr(self.ai_models, 'qwen_model') or self.ai_models.qwen_model is None:
                print("[🤖 Qwen 모델 로딩 시작]")
                self.ai_models.load_qwen_model()
            
            prompt = f"""한국어 명령에서 교정할 텍스트만 추출하세요.
형식: text|교정할텍스트

규칙:
- 교정해줘, 맞춤법, 문법 등의 명령어는 제거
- 실제 교정이 필요한 텍스트만 추출

예시:
"안녕하세요. 제가 오늘 회의에 참석못할것 같습니다 교정해주세요" → text|안녕하세요. 제가 오늘 회의에 참석못할것 같습니다
"'I can't attend meeting today' 교정해줘" → text|I can't attend meeting today
"맞춤법 검사: 안녕하새요" → text|안녕하새요
"문법 체크해줘 오늘 저녁에 뭐 먹을까요" → text|오늘 저녁에 뭐 먹을까요

입력: "{user_input}"
결과:"""
            
            # 토큰화
            inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt")
            
            # 생성
            import torch
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=100,  # 교정할 텍스트는 길 수 있음
                    do_sample=False,
                    temperature=0.1,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                    pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                )
            
            generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 응답 추출
            if "결과:" in generated_text:
                qwen_response = generated_text.split("결과:")[-1].strip()
            else:
                qwen_response = generated_text[len(prompt):].strip()
            
            print(f"[🤖 Qwen 응답] {qwen_response}")
            
            # 응답 파싱: "text|value" 형식
            lines = qwen_response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line and line.startswith('text|'):
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        extracted_text = parts[1].strip()
                        # 따옴표 제거
                        if extracted_text.startswith('"') and extracted_text.endswith('"'):
                            extracted_text = extracted_text[1:-1]
                        if extracted_text.startswith("'") and extracted_text.endswith("'"):
                            extracted_text = extracted_text[1:-1]
                        
                        print(f"[✅ 교정 텍스트 추출 성공] '{extracted_text}'")
                        return extracted_text
            
            print(f"[❌ 교정 텍스트 추출 실패] 파싱할 수 없는 응답: '{qwen_response}'")
            return None
            
        except Exception as e:
            print(f"[❗ Qwen 교정 텍스트 추출 오류] {str(e)}")
            return None

    def _extract_keyword_with_qwen(self, user_input):
        """Qwen을 사용하여 검색 키워드만 정확히 추출"""
        try:
            print(f"[🤖 Qwen 키워드 추출] '{user_input}'")
            
            # Qwen 모델 로딩
            if not hasattr(self.ai_models, 'qwen_model') or self.ai_models.qwen_model is None:
                print("[🤖 Qwen 모델 로딩 시작]")
                self.ai_models.load_qwen_model()
            
            prompt = f"""한국어 명령에서 검색 키워드만 추출하세요.
형식: keyword|추출된키워드

규칙:
- 메일, 이메일, 찾아줘, 검색, 보여줘는 반드시 제거
- 핵심 검색어만 남기기
- 영어 단어도 그대로 유지

예시:
"회의 관련 메일 검색해줘" → keyword|회의 관련
"ngrok 이메일을 찾아줘" → keyword|ngrok
"notion team 이메일을 찾아줘" → keyword|notion team  
"zoom 관련 메일" → keyword|zoom
"프로젝트 업데이트 찾아줘" → keyword|프로젝트 업데이트
"ChatGPT 메일 보여줘" → keyword|ChatGPT

입력: "{user_input}"
결과:"""
            
            # 토큰화
            inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt")
            
            # 생성
            import torch
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=30,  # 키워드는 짧음
                    do_sample=False,
                    temperature=0.1,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                    pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                )
            
            generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 응답 추출
            if "결과:" in generated_text:
                qwen_response = generated_text.split("결과:")[-1].strip()
            else:
                qwen_response = generated_text[len(prompt):].strip()
            
            print(f"[🤖 Qwen 응답] {qwen_response}")
            
            # 응답 파싱: "keyword|value" 형식
            lines = qwen_response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line and line.startswith('keyword|'):
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        extracted_keyword = parts[1].strip()
                        print(f"[✅ 키워드 추출 성공] '{extracted_keyword}'")
                        return extracted_keyword
            
            print(f"[❌ 키워드 추출 실패] 파싱할 수 없는 응답: '{qwen_response}'")
            return None
            
        except Exception as e:
            print(f"[❗ Qwen 키워드 추출 오류] {str(e)}")
            return None

    def _extract_date_with_qwen(self, user_input):
        """Qwen을 사용하여 날짜 정보 추출"""
        try:
            print(f"[🤖 Qwen 날짜 추출] '{user_input}'")
            
            # Qwen 모델 로딩
            if not hasattr(self.ai_models, 'qwen_model') or self.ai_models.qwen_model is None:
                print("[🤖 Qwen 모델 로딩 시작]")
                self.ai_models.load_qwen_model()
            
            prompt = f"""한국어 명령에서 날짜 정보를 추출하세요.
형식: date|날짜타입

날짜 타입:
- today: 오늘
- yesterday: 어제
- this_week: 이번주, 이번 주
- last_week: 지난주, 지난 주  
- this_month: 이번달, 이번 달
- last_month: 지난달, 지난 달
- none: 날짜 없음

예시:
"오늘 메일 찾아줘" → date|today
"어제 받은 메일" → date|yesterday
"지난주 회의록" → date|last_week
"이번달 보고서" → date|this_month
"회의 메일 찾아줘" → date|none

입력: "{user_input}"
결과:"""
            
            # 토큰화
            inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt")
            
            # 생성
            import torch
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=15,  # 날짜 정보는 매우 짧음
                    do_sample=False,
                    temperature=0.1,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                    pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                )
            
            generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 응답 추출
            if "결과:" in generated_text:
                qwen_response = generated_text.split("결과:")[-1].strip()
            else:
                qwen_response = generated_text[len(prompt):].strip()
            
            print(f"[🤖 Qwen 응답] {qwen_response}")
            
            # 응답 파싱: "date|value" 형식
            lines = qwen_response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line and line.startswith('date|'):
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        date_type = parts[1].strip()
                        if date_type != "none":
                            print(f"[✅ 날짜 추출 성공] '{date_type}'")
                            return date_type
            
            print(f"[📅 날짜 없음] 날짜 키워드가 없습니다")
            return None
            
        except Exception as e:
            print(f"[❗ Qwen 날짜 추출 오류] {str(e)}")
            return None

    def _extract_limit_with_qwen(self, user_input):
        """Qwen을 사용하여 개수 제한 정보 추출"""
        try:
            print(f"[🤖 Qwen 개수 추출] '{user_input}'")
            
            # Qwen 모델 로딩
            if not hasattr(self.ai_models, 'qwen_model') or self.ai_models.qwen_model is None:
                print("[🤖 Qwen 모델 로딩 시작]")
                self.ai_models.load_qwen_model()
            
            prompt = f"""한국어 명령에서 개수 제한을 추출하세요.
형식: limit|숫자

규칙:
- 숫자+개 패턴 찾기
- 개수 제한이 없으면 none

예시:
"메일 5개만 찾아줘" → limit|5
"최신 메일 10개 보여줘" → limit|10
"3개만 표시해줘" → limit|3
"회의 메일 찾아줘" → limit|none
"상위 20개 메일" → limit|20

입력: "{user_input}"
결과:"""
            
            # 토큰화
            inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt")
            
            # 생성
            import torch
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=15,  # 개수 정보는 매우 짧음
                    do_sample=False,
                    temperature=0.1,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                    pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                )
            
            generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 응답 추출
            if "결과:" in generated_text:
                qwen_response = generated_text.split("결과:")[-1].strip()
            else:
                qwen_response = generated_text[len(prompt):].strip()
            
            print(f"[🤖 Qwen 응답] {qwen_response}")
            
            # 응답 파싱: "limit|number" 형식
            lines = qwen_response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line and line.startswith('limit|'):
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        limit_str = parts[1].strip()
                        if limit_str != "none" and limit_str.isdigit():
                            limit_num = int(limit_str)
                            print(f"[✅ 개수 추출 성공] {limit_num}개")
                            return limit_num
            
            print(f"[🔢 개수 없음] 개수 제한이 없습니다")
            return None
            
        except Exception as e:
            print(f"[❗ Qwen 개수 추출 오류] {str(e)}")
            return None

    def _convert_date_type_to_filter(self, date_type):
        """Qwen에서 추출한 날짜 타입을 필터로 변환"""
        from datetime import datetime, timedelta
        
        today = datetime.now()
        
        if date_type == "today":
            start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
            return {
                'type': 'today',
                'start_date': start_date,
                'end_date': end_date
            }
        elif date_type == "yesterday":
            yesterday = today - timedelta(days=1)
            start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            return {
                'type': 'yesterday',
                'start_date': start_date,
                'end_date': end_date
            }
        elif date_type == "this_week":
            days_since_monday = today.weekday()
            this_monday = today - timedelta(days=days_since_monday)
            start_date = this_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
            return {
                'type': 'this_week',
                'start_date': start_date,
                'end_date': end_date
            }
        elif date_type == "last_week":
            days_since_monday = today.weekday()
            last_monday = today - timedelta(days=days_since_monday + 7)
            last_sunday = last_monday + timedelta(days=6)
            start_date = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = last_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
            return {
                'type': 'last_week',
                'start_date': start_date,
                'end_date': end_date
            }
        elif date_type == "this_month":
            first_day_this_month = today.replace(day=1)
            start_date = first_day_this_month.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
            return {
                'type': 'this_month',
                'start_date': start_date,
                'end_date': end_date
            }
        elif date_type == "last_month":
            first_day_this_month = today.replace(day=1)
            last_day_last_month = first_day_this_month - timedelta(days=1)
            first_day_last_month = last_day_last_month.replace(day=1)
            start_date = first_day_last_month.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = last_day_last_month.replace(hour=23, minute=59, second=59, microsecond=999999)
            return {
                'type': 'last_month',
                'start_date': start_date,
                'end_date': end_date
            }
        
        return None

#     def _classify_intent_with_qwen(self, user_input):
#         """Qwen을 사용하여 사용자 의도 분류"""
#         try:
#             print(f"[🤖 Qwen Intent 분류] '{user_input}'")
            
#             # Qwen 모델 로딩
#             if not hasattr(self.ai_models, 'qwen_model') or self.ai_models.qwen_model is None:
#                 print("[🤖 Qwen 모델 로딩 시작]")
#                 self.ai_models.load_qwen_model()
            
#             prompt = f"""한국어 명령의 의도를 분류하세요.
# 형식: intent|의도타입

# 의도 타입:
# - grammar_correction: 문법/맞춤법 교정 요청
# - email_search: 키워드로 메일 검색
# - person_search: 특정 사람의 메일 검색  
# - email_statistics: 메일 개수/통계 조회
# - settings_control: 앱 설정 변경
# - generate_ai_reply: AI 답장 생성

# 예시:
# "안녕하세요 교정해주세요" → intent|grammar_correction
# "회의 관련 메일 찾아줘" → intent|email_search
# "notion team 이메일 찾아줘" → intent|email_search
# "김철수님 메일 보여줘" → intent|person_search
# "오늘 메일 몇 개?" → intent|email_statistics
# "폰트 크기 18로 바꿔줘" → intent|settings_control
# "답장 생성해줘" → intent|generate_ai_reply

# 입력: "{user_input}"
# 결과:"""
            
#             # 토큰화
#             inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt")
            
#             # 생성
#             import torch
#             with torch.no_grad():
#                 outputs = self.ai_models.qwen_model.generate(
#                     inputs.input_ids,
#                     max_new_tokens=20,  # Intent는 짧음
#                     do_sample=False,
#                     temperature=0.1,
#                     eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
#                     pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
#                 )
            
#             generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # 응답 추출
#             if "결과:" in generated_text:
#                 qwen_response = generated_text.split("결과:")[-1].strip()
#             else:
#                 qwen_response = generated_text[len(prompt):].strip()
            
#             print(f"[🤖 Qwen 응답] {qwen_response}")
            
#             # 응답 파싱: "intent|type" 형식
#             lines = qwen_response.strip().split('\n')
#             for line in lines:
#                 line = line.strip()
#                 if '|' in line and line.startswith('intent|'):
#                     parts = line.split('|', 1)
#                     if len(parts) == 2:
#                         intent_type = parts[1].strip()
#                         valid_intents = [
#                             'grammar_correction', 'email_search', 'person_search',
#                             'email_statistics', 'settings_control', 'generate_ai_reply'
#                         ]
#                         if intent_type in valid_intents:
#                             print(f"[✅ Intent 분류 성공] '{intent_type}'")
#                             return {
#                                 'action': intent_type,
#                                 'confidence': 0.9,  # Qwen은 높은 신뢰도
#                                 'method': 'qwen_intent',
#                                 'detailed_intent': f'{intent_type} classified by Qwen'
#                             }
            
#             print(f"[❌ Intent 분류 실패] 파싱할 수 없는 응답: '{qwen_response}'")
#             return None
            
#         except Exception as e:
#             print(f"[❗ Qwen Intent 분류 오류] {str(e)}")
#             return None

#0826 수정
    def _classify_intent_with_qwen(self, user_input):
        """Qwen 기반 정확한 의도 분석"""
        # 새로운 단순 형식 파싱: "action, keyword"
        try:
            # NPU는 프롬프트를 포함하지 않는 "응답만" 반환한다고 가정
            qwen_response = genie_analyze_intent(user_input)
            # 앵커 정리
            #qwen_response = npu_out.split("결과:", 1)[-1].strip() if "결과:" in npu_out else npu_out.strip()

            # 디버그 (로그용 전체 문자열)
            # debug_prompt = qwen_prompt_command(user_input)
            # generated_text = _ensure_utf8(debug_prompt) + qwen_response
            # print(f"[🔍 디버그] 전체 생성된 텍스트 길이: {len(generated_text)}")
            # print(f"[🔍 디버그] 전체 생성 텍스트 일부: {generated_text[:200]}...")
            print(f"[🤖 NPU 원본 응답] {qwen_response}")

            # 응답 파싱: "intent|type" 형식
            lines = qwen_response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line and line.startswith('intent|'):
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        intent_type = parts[1].strip()
                        valid_intents = [
                            'grammar_correction', 'email_search', 'person_search',
                            'email_statistics', 'settings_control', 'generate_ai_reply'
                        ]
                        if intent_type in valid_intents:
                            print(f"[✅ Intent 분류 성공] '{intent_type}'")
                            return {
                                'action': intent_type,
                                'confidence': 0.9,  # Qwen은 높은 신뢰도
                                'method': 'qwen_intent',
                                'detailed_intent': f'{intent_type} classified by Qwen'
                            }

            # NPU 응답 파싱 실패 → 내부 키워드 폴백
            #print("[🔄 NPU 폴백] 응답 파싱 실패, 키워드 기반 분석으로 전환")
            #return self._parse_qwen_response_fallback(user_input, qwen_response)

        except Exception as ge:
            print(f"[⚠️ NPU(Genie) 실패] {ge} → HF(Qwen) 경로로 폴백")
        try:
            print(f"[🤖 Qwen Intent 분류] '{user_input}'")

            # Qwen 모델 로딩
            if not hasattr(self.ai_models, 'qwen_model') or self.ai_models.qwen_model is None:
                print("[🤖 Qwen 모델 로딩 시작]")
                self.ai_models.load_qwen_model()

            prompt = f"""한국어 명령의 의도를 분류하세요.
        형식: intent|의도타입

        의도 타입:
        - grammar_correction: 문법/맞춤법 교정 요청
        - email_search: 키워드로 메일 검색
        - person_search: 특정 사람의 메일 검색  
        - email_statistics: 메일 개수/통계 조회
        - settings_control: 앱 설정 변경
        - generate_ai_reply: AI 답장 생성

        예시:
        "안녕하세요 교정해주세요" → intent|grammar_correction
        "회의 관련 메일 찾아줘" → intent|email_search
        "notion team 이메일 찾아줘" → intent|email_search
        "김철수님 메일 보여줘" → intent|person_search
        "오늘 메일 몇 개?" → intent|email_statistics
        "폰트 크기 18로 바꿔줘" → intent|settings_control
        "답장 생성해줘" → intent|generate_ai_reply

        입력: "{user_input}"
        결과:"""

            # 토큰화
            inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt")

            # 생성
            import torch
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=20,  # Intent는 짧음
                    do_sample=False,
                    temperature=0.1,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                    pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                )

            generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 응답 추출
            if "결과:" in generated_text:
                qwen_response = generated_text.split("결과:")[-1].strip()
            else:
                qwen_response = generated_text[len(prompt):].strip()

            print(f"[🤖 Qwen 응답] {qwen_response}")

            # 응답 파싱: "intent|type" 형식
            lines = qwen_response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line and line.startswith('intent|'):
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        intent_type = parts[1].strip()
                        valid_intents = [
                            'grammar_correction', 'email_search', 'person_search',
                            'email_statistics', 'settings_control', 'generate_ai_reply'
                        ]
                        if intent_type in valid_intents:
                            print(f"[✅ Intent 분류 성공] '{intent_type}'")
                            return {
                                'action': intent_type,
                                'confidence': 0.9,  # Qwen은 높은 신뢰도
                                'method': 'qwen_intent',
                                'detailed_intent': f'{intent_type} classified by Qwen'
                            }

            print(f"[❌ Intent 분류 실패] 파싱할 수 없는 응답: '{qwen_response}'")
            return None

        except Exception as e:
            print(f"[❗ Qwen Intent 분류 오류] {str(e)}")
            return None
        #0826 끝

    def _extract_search_target_with_qwen(self, text):
        """Qwen을 이용하여 검색 대상 추출"""
        # Qwen 모델이 로딩되지 않았다면 로딩 시도
        if not self.ai_models.load_qwen_model():
            print("[⚠️ Qwen 모델 없음 - 간단 추출 사용]")
            words = text.split()
            return " ".join(words[-2:]) if len(words) >= 2 else text
        
        try:
            import torch
            prompt = (
                "<|im_start|>system\nYou are an email assistant. "
                "Your job is to extract the email address or name the user is referring to. "
                "You must always respond in the format: The user is referring to ... \n"
                "<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            
            inputs = self.ai_models.qwen_tokenizer(prompt, return_tensors="pt").to(self.ai_models.qwen_model.device)
            
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id
                )
            
            decoded_output = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # "assistant" 이후 텍스트만 가져옴
            if "assistant" in decoded_output:
                after_assistant = decoded_output.split("assistant")[-1].strip()
                prefix = "The user is referring to "
                if prefix in after_assistant:
                    result = after_assistant.split(prefix)[-1].strip().rstrip(".").strip('"')
                    return result
            
            return text
            
        except Exception as e:
            print(f"[⚠️ Qwen 추출 오류] {str(e)}")
            # 오류 시 간단한 키워드 추출로 fallback
            words = text.split()
            return " ".join(words[-2:]) if len(words) >= 2 else text
    
    def _handle_email_statistics(self, user_input, user_email, app_password):
        """이메일 통계 처리"""
        try:
            from models.tables import Mail
            from models.db import db
            from datetime import datetime, timedelta
            import time
            
            start_time = time.time()
            
            print(f"\n{'='*50}")
            print(f"[📊 통계 요청 시작] 사용자: {user_email}")
            print(f"[📝 통계 명령어] '{user_input}'")
            print(f"{'='*50}")
            
            user_input_lower = user_input.lower()
            
            # 오늘 날짜
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            this_week_start = today - timedelta(days=today.weekday())
            this_month_start = today.replace(day=1)
            
            print(f"[📅 날짜 계산 완료]")
            print(f"  • 오늘: {today}")
            print(f"  • 어제: {yesterday}")
            print(f"  • 이번주 시작: {this_week_start}")
            print(f"  • 이번달 시작: {this_month_start}")
            
            # 기본 쿼리
            base_query = Mail.query.filter_by(user_email=user_email)
            print(f"[🗄️ DB 쿼리] 사용자 '{user_email}' 메일 조회 준비")
            
            # 통계 결과 저장
            stats_result = "📊 **이메일 통계**\n\n"
            
            # 1. 오늘 관련 통계
            if any(keyword in user_input_lower for keyword in ["오늘", "today"]):
                print(f"[🎯 통계 유형] 오늘 메일 통계 요청")
                
                print(f"[🔍 DB 조회] 오늘 받은메일 개수 계산 중...")
                today_inbox = base_query.filter(
                    db.func.date(Mail.date) == today,
                    Mail.mail_type == 'inbox'
                ).count()
                
                print(f"[🔍 DB 조회] 오늘 보낸메일 개수 계산 중...")
                today_sent = base_query.filter(
                    db.func.date(Mail.date) == today,
                    Mail.mail_type == 'sent'
                ).count()
                
                print(f"[📊 계산 결과] 오늘 받은메일: {today_inbox}개, 보낸메일: {today_sent}개, 총 {today_inbox + today_sent}개")
                
                stats_result += f"📅 **오늘 ({today.strftime('%Y-%m-%d')})**\n"
                stats_result += f"📥 받은 메일: **{today_inbox}개**\n"
                stats_result += f"📤 보낸 메일: **{today_sent}개**\n"
                stats_result += f"📊 총합: **{today_inbox + today_sent}개**\n\n"
            
            # 2. 어제 관련 통계
            elif any(keyword in user_input_lower for keyword in ["어제", "yesterday"]):
                print(f"[🎯 통계 유형] 어제 메일 통계 요청")
                
                print(f"[🔍 DB 조회] 어제 받은메일 개수 계산 중...")
                yesterday_inbox = base_query.filter(
                    db.func.date(Mail.date) == yesterday,
                    Mail.mail_type == 'inbox'
                ).count()
                
                print(f"[🔍 DB 조회] 어제 보낸메일 개수 계산 중...")
                yesterday_sent = base_query.filter(
                    db.func.date(Mail.date) == yesterday,
                    Mail.mail_type == 'sent'
                ).count()
                
                print(f"[📊 계산 결과] 어제 받은메일: {yesterday_inbox}개, 보낸메일: {yesterday_sent}개, 총 {yesterday_inbox + yesterday_sent}개")
                
                stats_result += f"📅 **어제 ({yesterday.strftime('%Y-%m-%d')})**\n"
                stats_result += f"📥 받은 메일: **{yesterday_inbox}개**\n"
                stats_result += f"📤 보낸 메일: **{yesterday_sent}개**\n"
                stats_result += f"📊 총합: **{yesterday_inbox + yesterday_sent}개**\n\n"
            
            # 3. 이번주 관련 통계
            elif any(keyword in user_input_lower for keyword in ["이번주", "이번 주", "this week"]):
                print(f"[🎯 통계 유형] 이번주 메일 통계 요청")
                
                print(f"[🔍 DB 조회] 이번주 받은메일 개수 계산 중...")
                week_inbox = base_query.filter(
                    Mail.date >= this_week_start,
                    Mail.mail_type == 'inbox'
                ).count()
                
                print(f"[🔍 DB 조회] 이번주 보낸메일 개수 계산 중...")
                week_sent = base_query.filter(
                    Mail.date >= this_week_start,
                    Mail.mail_type == 'sent'
                ).count()
                
                print(f"[📊 계산 결과] 이번주 받은메일: {week_inbox}개, 보낸메일: {week_sent}개, 총 {week_inbox + week_sent}개")
                
                stats_result += f"📅 **이번주 ({this_week_start.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')})**\n"
                stats_result += f"📥 받은 메일: **{week_inbox}개**\n"
                stats_result += f"📤 보낸 메일: **{week_sent}개**\n"
                stats_result += f"📊 총합: **{week_inbox + week_sent}개**\n\n"
            
            # 4. 이번달 관련 통계
            elif any(keyword in user_input_lower for keyword in ["이번달", "이번 달", "this month"]):
                print(f"[🎯 통계 유형] 이번달 메일 통계 요청")
                
                print(f"[🔍 DB 조회] 이번달 받은메일 개수 계산 중...")
                month_inbox = base_query.filter(
                    Mail.date >= this_month_start,
                    Mail.mail_type == 'inbox'
                ).count()
                
                print(f"[🔍 DB 조회] 이번달 보낸메일 개수 계산 중...")
                month_sent = base_query.filter(
                    Mail.date >= this_month_start,
                    Mail.mail_type == 'sent'
                ).count()
                
                print(f"[📊 계산 결과] 이번달 받은메일: {month_inbox}개, 보낸메일: {month_sent}개, 총 {month_inbox + month_sent}개")
                
                stats_result += f"📅 **이번달 ({this_month_start.strftime('%Y-%m')})**\n"
                stats_result += f"📥 받은 메일: **{month_inbox}개**\n"
                stats_result += f"📤 보낸 메일: **{month_sent}개**\n"
                stats_result += f"📊 총합: **{month_inbox + month_sent}개**\n\n"
            
            # 5. 전체 통계 (기본값)
            else:
                print(f"[🎯 통계 유형] 전체 메일 통계 요청")
                
                print(f"[🔍 DB 조회] 전체 받은메일 개수 계산 중...")
                total_inbox = base_query.filter_by(mail_type='inbox').count()
                
                print(f"[🔍 DB 조회] 전체 보낸메일 개수 계산 중...")
                total_sent = base_query.filter_by(mail_type='sent').count()
                
                print(f"[🔍 DB 조회] 최근 메일 정보 조회 중...")
                # 최근 메일 날짜
                latest_mail = base_query.order_by(Mail.date.desc()).first()
                oldest_mail = base_query.order_by(Mail.date.asc()).first()
                
                print(f"[📊 계산 결과] 전체 받은메일: {total_inbox}개, 보낸메일: {total_sent}개, 총 {total_inbox + total_sent}개")
                if latest_mail:
                    print(f"[📅 최신 메일] {latest_mail.date.strftime('%Y-%m-%d %H:%M')}")
                if oldest_mail:
                    print(f"[📅 가장 오래된 메일] {oldest_mail.date.strftime('%Y-%m-%d %H:%M')}")
                
                stats_result += f"📊 **전체 이메일 통계**\n"
                stats_result += f"📥 총 받은 메일: **{total_inbox}개**\n"
                stats_result += f"📤 총 보낸 메일: **{total_sent}개**\n"
                stats_result += f"📈 전체 총합: **{total_inbox + total_sent}개**\n\n"
                
                if latest_mail:
                    stats_result += f"📅 **최근 메일**: {latest_mail.date.strftime('%Y-%m-%d %H:%M')}\n"
                if oldest_mail:
                    stats_result += f"📅 **가장 오래된 메일**: {oldest_mail.date.strftime('%Y-%m-%d %H:%M')}\n\n"
                
                # 받은메일 vs 보낸메일 비율
                if total_inbox + total_sent > 0:
                    inbox_ratio = (total_inbox / (total_inbox + total_sent)) * 100
                    sent_ratio = (total_sent / (total_inbox + total_sent)) * 100
                    print(f"[📈 비율 계산] 받은메일: {inbox_ratio:.1f}%, 보낸메일: {sent_ratio:.1f}%")
                    stats_result += f"📊 **비율**\n"
                    stats_result += f"📥 받은메일: {inbox_ratio:.1f}%\n"
                    stats_result += f"📤 보낸메일: {sent_ratio:.1f}%\n\n"
            
            # 추가 정보
            stats_result += "💡 **더 자세한 통계**\n"
            stats_result += "• '오늘 메일 몇 개?' - 오늘 통계\n"
            stats_result += "• '이번주 메일 개수' - 주간 통계\n"
            stats_result += "• '이번달 메일 통계' - 월간 통계\n"
            stats_result += "• '어제 메일 몇 개?' - 어제 통계"
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            print(f"[⏱️ 통계 처리 완료] 총 소요시간: {processing_time:.3f}초")
            print(f"[✅ 통계 응답 생성 완료] 응답 길이: {len(stats_result)}자")
            print(f"{'='*50}\n")
            
            return stats_result
            
        except Exception as e:
            print(f"[❗통계 처리 오류] {str(e)}")
            print(f"{'='*50}\n")
            return "❌ 통계 처리 중 오류가 발생했습니다."

    def _handle_date_search(self, user_input, user_email, app_password):
        """날짜별 검색 전용 핸들러"""
        print(f"[📅 날짜별 검색] 입력: '{user_input}'")
        
        # 날짜 필터를 우선적으로 파싱
        date_filter = self._parse_date_keywords(user_input)
        
        if not date_filter:
            # 날짜 키워드가 없으면 오늘 기본값
            from datetime import datetime
            today = datetime.now()
            date_filter = {
                'type': 'today',
                'start_date': today.replace(hour=0, minute=0, second=0, microsecond=0),
                'end_date': today.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
            
        # 추가 필터 파싱
        limit_count = self._parse_limit_keywords(user_input)
        mail_type_filter = self._parse_mail_type_keywords(user_input)
        
        # 날짜 중심 검색 실행
        return self._execute_search_with_filters(
            user_email, app_password, user_input, 
            date_filter, limit_count, mail_type_filter,
            focus="date"
        )
    
    def _handle_limit_search(self, user_input, user_email, app_password):
        """개수 제한 검색 전용 핸들러"""
        print(f"[🔢 개수 제한 검색] 입력: '{user_input}'")
        
        # Qwen으로 개수 제한 추출
        qwen_limit = self._extract_limit_with_qwen(user_input)
        limit_count = qwen_limit if qwen_limit else 5  # 기본 5개
        
        # Qwen으로 날짜 필터 추출  
        qwen_date = self._extract_date_with_qwen(user_input)
        date_filter = self._convert_date_type_to_filter(qwen_date) if qwen_date else None
        
        mail_type_filter = self._parse_mail_type_keywords(user_input)
        
        # 개수 제한 검색 (키워드 없이 최신 메일만)
        print(f"[🔢 제한 검색 실행] 개수: {limit_count}개")
        
        found_emails = self._search_emails_in_db(
            user_email, 
            search_keywords="",  # 키워드 없음
            max_results=limit_count,
            date_filter=date_filter,
            mail_type_filter=mail_type_filter,
            limit_count=limit_count
        )
        
        if found_emails:
            result = f"📬 **최신 메일 {limit_count}개**\n\n"
            for i, mail_info in enumerate(found_emails, 1):
                result += f"**{i}. {mail_info['subject']}**\n"
                result += f"👤 {mail_info['from']}\n"
                result += f"📅 {mail_info['date']}\n\n"
            return result
        else:
            return f"📭 메일이 없습니다."
    
    def _handle_type_search(self, user_input, user_email, app_password):
        """메일 타입별 검색 전용 핸들러"""
        print(f"[📧 타입별 검색] 입력: '{user_input}'")
        
        # 메일 타입을 우선적으로 파싱
        mail_type_filter = self._parse_mail_type_keywords(user_input)
        
        if not mail_type_filter:
            # 타입이 명시되지 않으면 받은메일 기본값
            mail_type_filter = "inbox"
            
        # 추가 필터 파싱
        date_filter = self._parse_date_keywords(user_input)
        limit_count = self._parse_limit_keywords(user_input)
        
        # 타입 중심 검색 실행
        return self._execute_search_with_filters(
            user_email, app_password, user_input,
            date_filter, limit_count, mail_type_filter,
            focus="type"
        )
    
    def _handle_complex_search(self, user_input, user_email, app_password):
        """복합 조건 검색 전용 핸들러"""
        print(f"[🔄 복합 검색] 입력: '{user_input}'")
        
        # 모든 필터를 동등하게 파싱
        date_filter = self._parse_date_keywords(user_input)
        limit_count = self._parse_limit_keywords(user_input)
        mail_type_filter = self._parse_mail_type_keywords(user_input)
        
        # 사람 이름도 추출
        person_name = self._extract_person_name(user_input)
        
        if person_name:
            # 사람별 + 복합 조건
            return self._handle_person_search_with_filters(
                user_input, user_email, app_password,
                person_name, date_filter, limit_count, mail_type_filter
            )
        else:
            # 일반 복합 검색
            return self._execute_search_with_filters(
                user_email, app_password, user_input,
                date_filter, limit_count, mail_type_filter,
                focus="complex"
            )
    
    def _execute_search_with_filters(self, user_email, app_password, user_input, 
                                    date_filter, limit_count, mail_type_filter, focus="general"):
        """필터를 적용한 검색 실행"""
        try:
            # 검색 키워드 추출
            search_keywords = self._clean_search_keywords(user_input, date_filter, limit_count, mail_type_filter)
            
            print(f"[🎯 {focus} 검색 실행] 키워드: '{search_keywords}', 날짜: {date_filter}, 개수: {limit_count}, 타입: {mail_type_filter}")
            
            # DB 검색 실행
            results = self._search_emails_in_db(
                user_email, 
                search_keywords,
                date_filter=date_filter,
                mail_type_filter=mail_type_filter,
                limit_count=limit_count
            )
            
            # 결과 포맷팅
            return self._format_search_results(results, search_keywords, focus)
            
        except Exception as e:
            print(f"[❗검색 실행 오류] {str(e)}")
            return f"검색 중 오류가 발생했습니다: {str(e)}"
    
    def _clean_search_keywords(self, user_input, date_filter, limit_count, mail_type_filter):
        """검색 키워드 정리 - 최소한의 처리만"""
        keywords = user_input
        
        # 오직 이미 추출된 필터 키워드만 제거 (중복 방지)
        if date_filter:
            date_keywords = ["오늘", "어제", "그제", "지난주", "이번주", "이번달", "지난달", "최근"]
            for kw in date_keywords:
                if kw in keywords:
                    keywords = keywords.replace(kw, "")
        
        if limit_count:
            import re
            # 숫자+개 패턴만 제거 (이미 limit_count로 추출됨)
            keywords = re.sub(r'\d+개\s*만?', '', keywords)
            if "최신" in keywords:
                keywords = keywords.replace("최신", "")
        
        if mail_type_filter:
            # 메일 타입 키워드만 제거 (이미 mail_type_filter로 추출됨)
            if "받은메일" in keywords:
                keywords = keywords.replace("받은메일", "")
            if "보낸메일" in keywords:
                keywords = keywords.replace("보낸메일", "")
        
        # 공백 정리만
        keywords = re.sub(r'\s+', ' ', keywords.strip())
        
        return keywords
    
    def _format_search_results(self, results, search_keywords, focus):
        """검색 결과 포맷팅"""
        if not results:
            return f"❌ '{search_keywords}'와 관련된 메일을 찾을 수 없습니다."
        
        # 포커스에 따른 제목 설정
        focus_titles = {
            "date": "📅 날짜별 검색 결과",
            "limit": "🔢 개수 제한 검색 결과", 
            "type": "📧 타입별 검색 결과",
            "complex": "🔄 복합 검색 결과",
            "general": "🔍 검색 결과"
        }
        
        response = f"{focus_titles.get(focus, '🔍 검색 결과')}\n\n"
        response += f"검색된 메일: {len(results)}개\n\n"
        
        for idx, mail in enumerate(results, 1):
            response += f"**{idx}. {mail['subject']}**\n"
            response += f"📤 발신자: {mail['from']}\n"
            response += f"📅 날짜: {mail['date']}\n"
            if mail.get('mail_type'):
                type_label = "받은메일" if mail['mail_type'] == 'inbox' else "보낸메일"
                response += f"📧 타입: {type_label}\n"
            response += f"📝 내용: {mail['body'][:100]}...\n\n"
        
        return response
    
    def _handle_person_search_with_filters(self, user_input, user_email, app_password, 
                                          person_name, date_filter, limit_count, mail_type_filter):
        """사람별 검색 + 추가 필터"""
        try:
            print(f"[👤 사람별 복합 검색] 사람: '{person_name}', 날짜: {date_filter}, 개수: {limit_count}, 타입: {mail_type_filter}")
            
            # 사람 이름으로 검색
            results = self._search_emails_in_db(
                user_email,
                person_name,
                date_filter=date_filter,
                mail_type_filter=mail_type_filter, 
                limit_count=limit_count
            )
            
            # 결과 포맷팅
            if not results:
                return f"❌ '{person_name}'님의 메일을 찾을 수 없습니다."
            
            response = f"👤 **{person_name}님 메일 검색 결과**\n\n"
            response += f"검색된 메일: {len(results)}개\n\n"
            
            for idx, mail in enumerate(results, 1):
                response += f"**{idx}. {mail['subject']}**\n"
                response += f"📅 날짜: {mail['date']}\n"
                response += f"📝 내용: {mail['body'][:100]}...\n\n"
            
            return response
            
        except Exception as e:
            print(f"[❗사람별 복합 검색 오류] {str(e)}")
            return f"검색 중 오류가 발생했습니다: {str(e)}"
    
    def _extract_person_name(self, user_input):
        """입력에서 사람 이름 추출"""
        # 님, 씨, 교수, 선생 등의 호칭이 있는 경우
        import re
        
        patterns = [
            r'([가-힣]+)(?:님|씨|교수|선생)',
            r'([a-zA-Z\s]+)(?:님|씨)',
            r'from\s+([a-zA-Z\s]+)',
            r'([가-힣]{2,4})(?:\s|의|에게|한테|로부터)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _handle_unknown_intent(self):
        """알 수 없는 의도 처리"""
        return """❓ 요청을 이해하지 못했습니다. 다른 표현을 시도해주세요.

🔧 **사용 가능한 기능들:**
• **문법/맞춤법 교정**: "이 문장 교정해주세요" / "correct this sentence"
• **이미지 생성**: "고양이 그림 그려줘" / "generate cat image"  
• **메일 검색**: "회의 관련 메일 찾아줘" / "find meeting emails"
• **사람별 메일**: "김철수님 메일 검색" / "search john@company.com emails"

💡 **Example / 예시:**
- 한국어: "안녕하세요. 제가 오늘 회의에 참석못할것 같습니다 교정해주세요"
- English: "correct the grammar: I can't attend meeting today"
- 혼합: "find 프로젝트 관련 emails" """

    

    def _search_emails_in_db(self, user_email, search_keywords, max_results=50, date_filter=None, mail_type_filter=None, limit_count=None):
        """DB에서 이메일 검색 (날짜/타입/개수 제한 지원)"""
        try:
            from models.tables import Mail
            from models.db import db
            import re
            
            print(f"[🔍 고급 검색 시작] 키워드: '{search_keywords}'")
            if date_filter:
                print(f"[📅 날짜 필터] {date_filter}")
            if mail_type_filter:
                print(f"[📧 타입 필터] {mail_type_filter}")
            if limit_count:
                print(f"[🔢 개수 제한] {limit_count}개")
            
            # 기본 쿼리 생성
            query = Mail.query.filter_by(user_email=user_email)
            
            # 날짜 필터 추가
            if date_filter:
                start_date = date_filter.get('start_date')
                end_date = date_filter.get('end_date')
                
                if start_date and end_date:
                    print(f"[📅 날짜 범위] {start_date} ~ {end_date}")
                    query = query.filter(
                        Mail.date >= start_date,
                        Mail.date <= end_date
                    )
                elif start_date:
                    print(f"[📅 시작 날짜] {start_date} 이후")
                    query = query.filter(Mail.date >= start_date)
                elif end_date:
                    print(f"[📅 종료 날짜] {end_date} 이전")
                    query = query.filter(Mail.date <= end_date)
            
            # 메일 타입 필터 추가
            if mail_type_filter:
                query = query.filter(Mail.mail_type == mail_type_filter)
            
            # 이메일 주소 패턴 확인
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_found = re.search(email_pattern, search_keywords)
            
            if email_found:
                # 이메일 주소로 검색 (발신자 기준)
                search_email = email_found.group()
                print(f"[🎯 이메일 주소 검색] {search_email}")
                
                query = query.filter(Mail.from_.contains(search_email))
                
            else:
                # 키워드로 제목/내용/발신자 검색
                print(f"[🎯 키워드 검색] {search_keywords}")
                
                query = query.filter(
                    db.or_(
                        Mail.subject.contains(search_keywords),
                        Mail.body.contains(search_keywords),
                        Mail.from_.contains(search_keywords),
                        Mail.summary.contains(search_keywords)
                    )
                )
            
            # 정렬 및 개수 제한
            final_limit = limit_count if limit_count else max_results
            db_results = query.order_by(Mail.date.desc()).limit(final_limit).all()
            
            # 결과를 기존 형태로 변환
            found_emails = []
            for mail in db_results:
                found_emails.append({
                    'id': mail.mail_id,
                    'subject': mail.subject[:60] + "..." if len(mail.subject) > 60 else mail.subject,
                    'from': mail.from_[:40] + "..." if len(mail.from_) > 40 else mail.from_,
                    'date': mail.date.strftime('%Y-%m-%d %H:%M:%S'),
                    'preview': mail.body[:200] + "..." if len(mail.body) > 200 else mail.body,
                    'classification': mail.classification,
                    'summary': mail.summary
                })
            
            print(f"[✅ 챗봇 DB 검색] {len(found_emails)}개 결과")
            return found_emails
            
        except Exception as e:
            print(f"[❗ 챗봇 DB 검색 실패] {str(e)}")
            return []
    
    def _parse_date_keywords(self, user_input):
        """사용자 입력에서 날짜 키워드 파싱"""
        try:
            from datetime import datetime, timedelta
            
            user_input_lower = user_input.lower()
            today = datetime.now()
            
            print(f"[📅 날짜 파싱] 입력: '{user_input}'")
            
            # 오늘
            if any(keyword in user_input_lower for keyword in ["오늘", "today"]):
                start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
                print(f"[📅 파싱 결과] 오늘: {start_date.date()}")
                return {
                    'type': 'today',
                    'start_date': start_date,
                    'end_date': end_date
                }
            
            # 어제
            elif any(keyword in user_input_lower for keyword in ["어제", "yesterday"]):
                yesterday = today - timedelta(days=1)
                start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
                print(f"[📅 파싱 결과] 어제: {start_date.date()}")
                return {
                    'type': 'yesterday',
                    'start_date': start_date,
                    'end_date': end_date
                }
            
            # 지난주
            elif any(keyword in user_input_lower for keyword in ["지난주", "last week"]):
                # 지난주 월요일부터 일요일까지
                days_since_monday = today.weekday()
                last_monday = today - timedelta(days=days_since_monday + 7)
                last_sunday = last_monday + timedelta(days=6)
                
                start_date = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = last_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
                print(f"[📅 파싱 결과] 지난주: {start_date.date()} ~ {end_date.date()}")
                return {
                    'type': 'last_week',
                    'start_date': start_date,
                    'end_date': end_date
                }
            
            # 이번주
            elif any(keyword in user_input_lower for keyword in ["이번주", "이번 주", "this week"]):
                days_since_monday = today.weekday()
                this_monday = today - timedelta(days=days_since_monday)
                
                start_date = this_monday.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
                print(f"[📅 파싱 결과] 이번주: {start_date.date()} ~ {end_date.date()}")
                return {
                    'type': 'this_week',
                    'start_date': start_date,
                    'end_date': end_date
                }
            
            # 지난달
            elif any(keyword in user_input_lower for keyword in ["지난달", "last month"]):
                # 지난달 1일부터 말일까지
                first_day_this_month = today.replace(day=1)
                last_day_last_month = first_day_this_month - timedelta(days=1)
                first_day_last_month = last_day_last_month.replace(day=1)
                
                start_date = first_day_last_month.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = last_day_last_month.replace(hour=23, minute=59, second=59, microsecond=999999)
                print(f"[📅 파싱 결과] 지난달: {start_date.date()} ~ {end_date.date()}")
                return {
                    'type': 'last_month',
                    'start_date': start_date,
                    'end_date': end_date
                }
            
            # 이번달
            elif any(keyword in user_input_lower for keyword in ["이번달", "이번 달", "this month"]):
                first_day_this_month = today.replace(day=1)
                
                start_date = first_day_this_month.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = today.replace(hour=23, minute=59, second=59, microsecond=999999)
                print(f"[📅 파싱 결과] 이번달: {start_date.date()} ~ {end_date.date()}")
                return {
                    'type': 'this_month',
                    'start_date': start_date,
                    'end_date': end_date
                }
            
            # 최근 N일
            import re
            recent_pattern = re.search(r'최근\s*(\d+)\s*일', user_input_lower)
            if recent_pattern:
                days = int(recent_pattern.group(1))
                start_date = today - timedelta(days=days)
                end_date = today
                print(f"[📅 파싱 결과] 최근 {days}일: {start_date.date()} ~ {end_date.date()}")
                return {
                    'type': f'recent_{days}_days',
                    'start_date': start_date,
                    'end_date': end_date
                }
            
            print(f"[📅 파싱 결과] 날짜 키워드 없음")
            return None
            
        except Exception as e:
            print(f"[❗날짜 파싱 오류] {str(e)}")
            return None
    
    def _parse_limit_keywords(self, user_input):
        """사용자 입력에서 개수 제한 키워드 파싱"""
        try:
            import re
            
            user_input_lower = user_input.lower()
            
            # "N개만", "N개", "N개까지" 패턴
            limit_patterns = [
                r'(\d+)\s*개\s*만',
                r'(\d+)\s*개\s*까지',
                r'(\d+)\s*개(?!\s*[이가는을])',  # "개" 뒤에 조사가 없는 경우
                r'최근\s*(\d+)\s*개',
                r'최신\s*(\d+)\s*개',
                r'처음\s*(\d+)\s*개',
                r'상위\s*(\d+)\s*개'
            ]
            
            for pattern in limit_patterns:
                match = re.search(pattern, user_input_lower)
                if match:
                    limit_count = int(match.group(1))
                    print(f"[🔢 개수 제한 파싱] {limit_count}개로 제한")
                    return limit_count
            
            print(f"[🔢 개수 제한 파싱] 제한 없음")
            return None
            
        except Exception as e:
            print(f"[❗개수 파싱 오류] {str(e)}")
            return None
    
    def _extract_settings_with_keywords(self, user_input):
        """키워드 기반으로 설정값을 추출"""
        try:
            print(f"[🔍 키워드 설정 추출] 입력: '{user_input}'")
            
            user_lower = user_input.lower()
            
            # 테마 설정 감지
            if any(keyword in user_lower for keyword in ['다크모드', '다크 모드', 'dark', '어두운']):
                return 'theme', 'dark'
            elif any(keyword in user_lower for keyword in ['라이트모드', '라이트 모드', 'light', '밝은', '기본']):
                return 'theme', 'light'
            elif any(keyword in user_lower for keyword in ['자동', 'auto', '시스템']):
                return 'theme', 'auto'
            
            # 폰트 크기 설정 감지
            if any(keyword in user_lower for keyword in ['폰트', '글자', '크기', 'font', 'size']):
                import re
                # 숫자 추출
                numbers = re.findall(r'\d+', user_input)
                if numbers:
                    size = int(numbers[0])
                    if 10 <= size <= 22:  # 유효한 폰트 크기 범위
                        return 'fontSize', f'{size}px'
            
            # 폰트 종류 설정 감지
            font_keywords = ['arial', 'helvetica', '나눔고딕', 'nanumgothic', '맑은고딕', 'malgun', 'times', '궁서']
            for font in font_keywords:
                if font in user_lower:
                    return 'fontFamily', font
            
            # Gmail 가져오기 개수 설정
            if any(keyword in user_lower for keyword in ['gmail', '지메일', '가져오기', '개수']):
                import re
                numbers = re.findall(r'\d+', user_input)
                if numbers:
                    count = int(numbers[0])
                    if 10 <= count <= 100:  # 유효한 범위
                        return 'gmailFetchCount', str(count)
            
            # 페이지당 아이템 개수 설정
            if any(keyword in user_lower for keyword in ['페이지', '목록', '아이템', '개수', '보여', '표시']):
                import re
                numbers = re.findall(r'\d+', user_input)
                if numbers:
                    count = int(numbers[0])
                    if 5 <= count <= 50:  # 유효한 범위
                        return 'itemsPerPage', str(count)
            
            # 발신자 이름 설정 - 더 정확한 패턴 매칭
            sender_keywords = ['발신자', '발신장', '보낸사람', '보낸이', '발송자', '송신자', 'sender', '보내는사람', '보내는이']
            name_keywords = ['이름', '명', '성명']
            action_keywords = ['바꿔', '변경', '설정', '수정', '고쳐', '바꾸', '변경해', '설정해']
            
            # 발신자 관련 키워드가 있는지 확인
            has_sender = any(keyword in user_lower for keyword in sender_keywords)
            has_name = any(keyword in user_lower for keyword in name_keywords)
            has_action = any(keyword in user_lower for keyword in action_keywords)
            
            # 발신자 + 이름 조합 또는 발신자 + 액션 조합이면 발신자 이름 설정
            if has_sender and (has_name or has_action):
                print(f"[📧 발신자 이름 설정 감지] 입력: '{user_input}'")
                return 'senderName_request', 'need_input'
            
            print(f"[❌ 설정 추출 실패] 인식할 수 없는 명령: '{user_input}'")
            return None, None
            
        except Exception as e:
            print(f"[❗ 키워드 설정 추출 오류] {str(e)}")
            return None, None

    def _handle_settings_control(self, user_input, user_email, details):
        """설정 변경 처리 (키워드 기반)"""
        try:
            import requests
            
            print(f"[⚙️ 설정 변경] 사용자 입력: '{user_input}'")
            print(f"[📋 세부사항] {details}")
            
            
            # 1. 키워드로 설정값 추출
            setting_type, setting_value = self._extract_settings_with_keywords(user_input)
            
            if not setting_type or not setting_value:
                return "❓ 설정 내용을 파악할 수 없습니다. 다시 말씀해주세요.\n\n예: '폰트 크기 18로', '다크모드로', 'Arial 폰트로'"
            
            # 2. 추출된 설정값으로 API 호출
            print(f"[🎯 설정 실행] {setting_type} → {setting_value}")
            
            # 테마 설정
            if setting_type == "theme":
                response = requests.put(
                    f'http://localhost:5001/api/settings/GENERAL/THEME/appearance',
                    json={
                        'email': user_email,
                        'value': setting_value
                    }
                )
                
                if response.status_code == 200:
                    # 설정 변경 완료 시 이벤트 발생
                    # 소켓 서버가 없으므로 이벤트 전송 불가
                    print(f"[⚠️ 소켓 서버 없음] 실시간 업데이트 불가 - UI 새로고침 필요")
                    
                    theme_names = {"dark": "다크 모드", "light": "라이트 모드", "auto": "자동 모드"}
                    return f"✅ 테마가 {theme_names.get(setting_value, setting_value)}로 변경되었습니다! 🎨"
                else:
                    return "❌ 테마 변경에 실패했습니다."
            
            # 폰트 크기 설정
            elif setting_type == "fontSize":
                # "18px" → 18 추출
                import re
                size_match = re.search(r'\d+', setting_value)
                if size_match:
                    size = int(size_match.group())
                    if 10 <= size <= 22:
                        response = requests.put(
                            f'http://localhost:5001/api/settings/GENERAL/WRITE/fontSize',
                            json={
                                'email': user_email,
                                'value': f'{size}px'
                            }
                        )
                        if response.status_code == 200:
                            # 설정 변경 완료 시 이벤트 발생
                            try:
                                import socketio
                                sio = socketio.SimpleClient()
                                sio.connect('http://localhost:5001')
                                sio.emit('settingsUpdated', {'email': user_email})
                                sio.disconnect()
                            except Exception as e:
                                print(f"[⚠️ 소켓 이벤트 전송 실패] {e}")
                            
                            return f"✅ 폰트 크기가 {size}px로 설정되었습니다! 🔤"
                        else:
                            return "❌ 폰트 크기 변경에 실패했습니다."
                    else:
                        return "⚠️ 폰트 크기는 10~22 사이여야 합니다."
                else:
                    return "❌ 올바른 폰트 크기 형식이 아닙니다."
            
            # 폰트 종류 설정  
            elif setting_type == "fontFamily":
                font_map = {
                    "Arial": "Arial",
                    "맑은고딕": "맑은 고딕", 
                    "돋움": "돋움",
                    "굴림": "굴림",
                    "바탕": "바탕",
                    "궁서": "궁서",
                    "Times": "Times New Roman",
                    "Helvetica": "Helvetica",
                    "Verdana": "Verdana", 
                    "Georgia": "Georgia",
                    "Courier": "Courier New",
                    "시스템기본": "system"
                }
                
                font_family = font_map.get(setting_value, setting_value)
                response = requests.put(
                    f'http://localhost:5001/api/settings/GENERAL/WRITE/fontFamily',
                    json={
                        'email': user_email,
                        'value': font_family
                    }
                )
                if response.status_code == 200:
                    # 설정 변경 완료 시 이벤트 발생
                    # 소켓 서버가 없으므로 이벤트 전송 불가
                    print(f"[⚠️ 소켓 서버 없음] 실시간 업데이트 불가 - UI 새로고침 필요")
                    
                    return f"✅ 폰트가 {font_family}로 변경되었습니다! 📝"
                else:
                    return "❌ 폰트 변경에 실패했습니다."
            
            # Gmail 가져오기 개수
            elif setting_type == "gmailFetchCount":
                count = int(setting_value)
                if 3 <= count <= 100:
                    response = requests.put(
                        f'http://localhost:5001/api/settings/GENERAL/READ/gmailFetchCount',
                        json={
                            'email': user_email,
                            'value': count
                        }
                    )
                    if response.status_code == 200:
                        # 설정 변경 완료 시 이벤트 발생
                        try:
                            import socketio
                            sio = socketio.SimpleClient()
                            sio.connect('http://localhost:5001')
                            sio.emit('settingsUpdated', {'email': user_email})
                            sio.disconnect()
                        except Exception as e:
                            print(f"[⚠️ 소켓 이벤트 전송 실패] {e}")
                        
                        return f"✅ Gmail 가져오기 개수가 {count}개로 설정되었습니다! 📧"
                    else:
                        return "❌ Gmail 개수 설정에 실패했습니다."
                else:
                    return "⚠️ Gmail 개수는 3~100 사이여야 합니다."
            
            # 페이지당 표시 개수
            elif setting_type == "itemsPerPage":
                size = int(setting_value)
                if 3 <= size <= 50:
                    response = requests.put(
                        f'http://localhost:5001/api/settings/GENERAL/READ/itemsPerPage',
                        json={
                            'email': user_email,
                            'value': size
                        }
                    )
                    if response.status_code == 200:
                        # 설정 변경 완료 시 이벤트 발생
                        try:
                            import socketio
                            sio = socketio.SimpleClient()
                            sio.connect('http://localhost:5001')
                            sio.emit('settingsUpdated', {'email': user_email})
                            sio.disconnect()
                        except Exception as e:
                            print(f"[⚠️ 소켓 이벤트 전송 실패] {e}")
                        
                        return f"✅ 페이지당 표시 개수가 {size}개로 설정되었습니다! 📄"
                    else:
                        return "❌ 페이지 설정에 실패했습니다."
                else:
                    return "⚠️ 페이지당 개수는 3~50 사이여야 합니다."
            
            # 발신자 이름 입력 요청
            elif setting_type == "senderName_request":
                # 임시 파일에 요청 상태 저장
                try:
                    import os
                    temp_dir = "user_sessions"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file = os.path.join(temp_dir, f"{user_email}_awaiting_name.txt")
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.write("waiting")
                    print(f"[💾 상태 저장] 발신자 이름 입력 대기 상태 저장")
                except Exception as e:
                    print(f"[⚠️ 상태 저장 실패] {e}")
                
                return """📧 **발신자 이름 설정**

원하는 발신자 이름을 입력해주세요.
예: 최수운, 김철수, John Smith"""

            # 실제 발신자 이름 설정
            elif setting_type == "senderName":
                print(f"[🔧 API 호출] PUT /api/settings/GENERAL/WRITE/senderName")
                print(f"[📤 요청 데이터] email: {user_email}, value: {setting_value}")
                
                response = requests.put(
                    f'http://localhost:5001/api/settings/GENERAL/WRITE/senderName',
                    json={
                        'email': user_email,
                        'value': setting_value
                    }
                )
                
                print(f"[📥 응답 상태] {response.status_code}")
                if response.status_code == 200:
                    response_data = response.json()
                    print(f"[📥 응답 데이터] {response_data}")
                    # 소켓 서버가 없으므로 이벤트 전송 불가
                    print(f"[⚠️ 소켓 서버 없음] Flask-SocketIO가 설정되지 않아 실시간 업데이트 불가")
                    print(f"[💡 해결방법] 설정 UI 페이지를 새로고침하면 변경사항이 반영됩니다.")
                    
                    # DB 값 확인 (검증용)
                    print(f"[🔍 DB 검증 시작] 설정이 실제로 저장되었는지 확인...")
                    try:
                        # 1초 대기 (DB 커밋 완료 대기)
                        import time
                        time.sleep(1)
                        
                        # 전체 WRITE 섹션 조회
                        verify_response = requests.get(
                            f'http://localhost:5001/api/settings/GENERAL/WRITE',
                            params={'email': user_email}
                        )
                        print(f"[🔍 DB 검증] GET 응답 코드: {verify_response.status_code}")
                        
                        if verify_response.status_code == 200:
                            verify_data = verify_response.json()
                            print(f"[🔍 DB 검증] 전체 응답: {verify_data}")
                            
                            settings_data = verify_data.get('settings', {})
                            actual_value = settings_data.get('senderName', 'N/A')
                            
                            print(f"[📊 DB 검증 결과]")
                            print(f"  - 요청한 값: '{setting_value}'")
                            print(f"  - 저장된 값: '{actual_value}'")
                            print(f"  - 전체 WRITE 설정: {settings_data}")
                            
                            if actual_value != setting_value:
                                print(f"[❌ DB 오류] 설정값 불일치! DB에 제대로 저장되지 않았습니다.")
                                print(f"[💡 원인] DB 커밋 실패 또는 세션 불일치 가능성")
                            else:
                                print(f"[✅ DB 성공] 설정값이 정상적으로 저장되었습니다!")
                                print(f"[💡 UI 문제] DB는 정상이므로 UI 새로고침이 필요합니다.")
                        else:
                            print(f"[❌ DB 검증 실패] API 응답 오류: {verify_response.text}")
                    except Exception as e:
                        print(f"[❌ DB 검증 실패] 예외 발생: {e}")
                    
                    return f"✅ 보내는 이름이 '{setting_value}'로 설정되었습니다! 👤"
                else:
                    response_data = response.json() if response.content else {}
                    error_msg = response_data.get('message', '알 수 없는 오류')
                    return f"❌ 보내는 이름 설정에 실패했습니다. 오류: {error_msg}"
                    
            else:
                return f"❓ 지원하지 않는 설정 타입입니다: {setting_type}"
                
        except Exception as e:
            print(f"[❗설정 변경 오류] {str(e)}")
            return f"❌ 설정 변경 중 오류가 발생했습니다: {str(e)}"
    
    def _parse_mail_type_keywords(self, user_input):
        """사용자 입력에서 메일 타입 키워드 파싱"""
        try:
            user_input_lower = user_input.lower()
            
            # 받은메일 키워드
            if any(keyword in user_input_lower for keyword in ["받은 메일", "받은메일", "수신", "inbox", "받은"]):
                print(f"[📧 타입 파싱] 받은메일만 검색")
                return 'inbox'
            
            # 보낸메일 키워드  
            elif any(keyword in user_input_lower for keyword in ["보낸 메일", "보낸메일", "발신", "sent", "보낸"]):
                print(f"[📧 타입 파싱] 보낸메일만 검색")
                return 'sent'
            
            print(f"[📧 타입 파싱] 모든 타입")
            return None
            
        except Exception as e:
            print(f"[❗타입 파싱 오류] {str(e)}")
            return None
    
    def _try_learned_pattern(self, user_email, user_input, app_password):
        """학습된 패턴에서 매칭 시도"""
        try:
            from models.tables import Chatbot
            
            # DB에서 사용자의 학습된 명령어들 조회
            print(f"[🔍 DB 조회] 사용자 '{user_email}'의 학습된 명령어 검색 중...")
            learned_commands = Chatbot.query.filter_by(user_email=user_email).all()
            
            print(f"[📊 DB 결과] 학습된 명령어 {len(learned_commands)}개 발견")
            if not learned_commands:
                print(f"[❌ 학습 데이터 없음] 처음 사용하는 명령어입니다")
                return None
            
            # 학습된 명령어 목록 출력
            for i, cmd in enumerate(learned_commands, 1):
                print(f"[📝 학습 #{i}] '{cmd.command}' -> {cmd.intent} (사용횟수: {cmd.use_count})")
            
            user_input_lower = user_input.lower()
            best_match = None
            best_similarity = 0.0
            
            print(f"[🔄 유사도 계산] 입력 명령어와 각 학습 데이터 비교 중...")
            
            # 각 학습된 명령어와 유사도 비교
            for i, learned in enumerate(learned_commands, 1):
                similarity = self._calculate_similarity_enhanced(user_input_lower, learned.command.lower())
                print(f"[📏 유사도 #{i}] '{learned.command}' vs '{user_input}' = {similarity:.3f} ({'✅ 임계값 통과' if similarity > 0.75 else '❌ 임계값 미달'})")
                
                if similarity > best_similarity and similarity > 0.75:  # 75% 이상
                    best_match = learned
                    best_similarity = similarity
            
            if best_match:
                print(f"[🎯 최고 매칭] '{best_match.command}' (유사도: {best_similarity:.3f})")
                print(f"[📈 사용 통계] 기존 {best_match.use_count}회 → {best_match.use_count + 1}회로 증가")
                
                # 사용 횟수 증가
                best_match.use_count += 1
                
                from models.db import db
                db.session.commit()
                
                # 학습된 intent로 실행 (원본 입력 포함)
                print(f"[🚀 학습 패턴 실행] intent='{best_match.intent}', keywords={best_match.get_keywords_dict()}")
                return self._execute_learned_intent(best_match.intent, best_match.get_keywords_dict(), user_email, app_password, original_input=user_input)
            else:
                print(f"[❌ 매칭 실패] 유사도 75% 이상인 학습 데이터 없음 (최고: {best_similarity:.3f})")
                
        except Exception as e:
            print(f"[❗ 학습 패턴 매칭 오류] {str(e)}")
            
        return None
    
    def _calculate_similarity_enhanced(self, cmd1, cmd2):
        """향상된 유사도 계산 (키워드 기반)"""
        
        # 핵심 키워드들 정의 (8개 타입 기반)
        key_words = [
            # sender
            "교수님", "회사", "학과", "선생님", "교직원", "naver", "google", "microsoft",
            # date  
            "오늘", "어제", "이번주", "지난주", "이번달", "지난달", "최근",
            # tag
            "중요", "스팸", "보안", "대학교",
            # attachment
            "첨부파일", "이미지", "pdf", "문서", "파일", "사진",
            # action
            "검색", "보여줘", "찾아줘", "작성", "답장", "삭제", "요약", "써줘",
            # content
            "과제", "회의", "공지", "영수증", "비밀번호", "로그인", "알림", "메일", "이메일",
            # settings (새로 추가)
            "폰트", "글꼴", "크기", "설정", "바꿔", "바꿔줘", "변경", "수정", "조절", "적용",
            "테마", "다크모드", "라이트모드", "Gmail", "개수", "페이지"
        ]
        
        common_keywords = 0
        total_keywords = 0
        
        for keyword in key_words:
            in_cmd1 = keyword in cmd1
            in_cmd2 = keyword in cmd2
            
            if in_cmd1 or in_cmd2:
                total_keywords += 1
                if in_cmd1 and in_cmd2:
                    common_keywords += 1
        
        # 키워드 기반 유사도
        keyword_similarity = common_keywords / total_keywords if total_keywords > 0 else 0
        
        # 기존 단어 기반 유사도도 같이 고려
        word_similarity = self._calculate_word_similarity(cmd1, cmd2)
        
        # 두 유사도의 평균 (키워드 기반을 더 중요하게)
        return keyword_similarity * 0.7 + word_similarity * 0.3
    
    def _calculate_word_similarity(self, cmd1, cmd2):
        """단어 기반 유사도 계산"""
        words1 = set(cmd1.split())
        words2 = set(cmd2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _execute_learned_intent(self, intent, keywords, user_email, app_password, original_input=None):
        """학습된 intent 실행"""
        
        print(f"[🎯 학습된 Intent 실행 시작] intent='{intent}'")
        print(f"[🏷️ 사용 키워드] {keywords}")
        print(f"[📝 원본 입력] '{original_input}'")
        
        # 기존 핸들러들을 그대로 활용
        if intent == "person_search":
            print(f"[👤 사람별 검색 실행] 키워드 기반 검색 대상 추출 중...")
            # keywords에서 검색 대상 추출
            search_target = keywords.get('sender', '') or keywords.get('content', '')
            print(f"[🎯 검색 대상] '{search_target}'")
            if search_target:
                # 기존 함수 활용하되, 검색 대상을 명시적으로 전달
                user_input_reconstructed = f"{search_target} 메일"
                print(f"[🔄 명령어 재구성] '{user_input_reconstructed}'")
                response = self._handle_person_search(user_input_reconstructed, user_email, app_password)
            else:
                print(f"[⚠️ 검색 대상 없음] 기본 사람 검색 실행")
                response = self._handle_person_search("", user_email, app_password)
                
        elif intent == "email_search":
            print(f"[🔍 일반 검색 실행] 키워드 기반 검색어 추출 중...")
            # 일반 검색
            search_keyword = keywords.get('content', '') or keywords.get('sender', '') or keywords.get('tag', '')
            print(f"[🎯 검색 키워드] '{search_keyword}'")
            if search_keyword:
                user_input_reconstructed = f"{search_keyword} 검색"
                print(f"[🔄 명령어 재구성] '{user_input_reconstructed}'")
                response = self._handle_general_search(user_input_reconstructed, user_email, app_password)
            else:
                print(f"[⚠️ 검색 키워드 없음] 기본 검색 실행")
                response = self._handle_general_search("", user_email, app_password)
                
        elif intent == "grammar_correction":
            print(f"[📝 문법 교정 실행] 기본 응답 제공")
            # 문법 교정 - 원본 명령어 필요하므로 기본 응답
            response = self._handle_grammar_correction("")
            
        elif intent == "image_generation":
            print(f"[🎨 이미지 생성 실행] 기본 응답 제공")
            response = self._handle_image_generation("")
            
        elif intent == "email_statistics":
            print(f"[📊 통계 실행] 키워드 기반 통계 생성")
            # 키워드에서 날짜 정보 추출하여 통계 생성
            date_keyword = keywords.get('date', '')
            if date_keyword:
                reconstructed_input = f"{date_keyword} 메일 몇 개"
                response = self._handle_email_statistics(reconstructed_input, user_email, app_password)
            else:
                response = self._handle_email_statistics("전체 통계", user_email, app_password)
            
        elif intent == "settings_control":
            print(f"[⚙️ 설정 변경 실행] 원본 명령어 사용")
            # 원본 입력을 그대로 사용 (재구성 하지 않음)
            if original_input:
                response = self._handle_settings_control(original_input, user_email, "")
            else:
                # 원본이 없으면 기본 메시지
                response = "설정 변경 명령을 다시 입력해주세요."
            
        else:
            print(f"[❌ 알 수 없는 Intent] '{intent}' 처리 불가")
            response = "학습된 패턴을 실행할 수 없습니다."
        
        print(f"[✅ 학습 패턴 실행 완료] 응답 생성됨")
        
        return {
            "response": response,
            "action": intent,
            "confidence": 1.0,  # 학습된 패턴은 높은 신뢰도
            "detected_intent": intent,
            "detection_method": "learned_pattern"
        }
    
    def _auto_save_learned_command(self, user_email, command, intent_result, response):
        """Qwen 분석 결과를 자동으로 학습 데이터로 저장"""
        try:
            from models.tables import Chatbot
            from models.db import db
            import json
            
            print(f"[💾 자동 저장] 명령어: '{command}'")
            print(f"[💾 자동 저장] 의도: {intent_result['action']} (신뢰도: {intent_result['confidence']:.3f})")
            
            # 기존에 동일한 명령어가 있는지 확인
            existing = Chatbot.query.filter_by(
                user_email=user_email,
                command=command
            ).first()
            
            if existing:
                print(f"[⚠️ 자동 저장] 이미 존재하는 명령어 - 업데이트")
                existing.intent = intent_result['action']
                existing.use_count += 1
                existing.keywords = json.dumps(self._extract_keywords_from_command(command), ensure_ascii=False)
                # response 필드는 Chatbot 모델에 없으므로 제거
            else:
                print(f"[✅ 자동 저장] 새로운 명령어 - 추가")
                new_command = Chatbot(
                    user_email=user_email,
                    command=command,
                    intent=intent_result['action'],
                    keywords=json.dumps(self._extract_keywords_from_command(command), ensure_ascii=False),
                    use_count=1
                )
                db.session.add(new_command)
            
            db.session.commit()
            print(f"[✅ 자동 저장] DB 저장 완료")
            
        except Exception as e:
            print(f"[❗ 자동 저장 오류] {str(e)}")
            try:
                db.session.rollback()
            except:
                pass

    def _save_learned_command(self, user_email, command, intent, response):
        """AI로 처리한 결과를 학습 데이터로 저장"""
        try:
            from models.tables import Chatbot
            from models.db import db
            import json
            
            print(f"[🔍 키워드 추출] 명령어에서 6개 타입 키워드 추출 중...")
            # 키워드 추출
            keywords = self._extract_keywords_from_command(command)
            print(f"[📝 추출된 키워드] {keywords}")
            
            print(f"[🔍 중복 검사] 기존 학습 데이터 확인 중...")
            # 이미 같은 명령어가 있는지 확인 (중복 방지)
            existing = Chatbot.query.filter_by(
                user_email=user_email,
                command=command
            ).first()
            
            if existing:
                # 기존 명령어의 사용 횟수만 증가
                print(f"[🔄 중복 데이터 발견] 기존 데이터 업데이트 진행")
                print(f"[📊 사용 횟수] {existing.use_count} → {existing.use_count + 1}")
                existing.use_count += 1
                print(f"[✅ 기존 학습 데이터 업데이트 완료]")
            else:
                # 새로운 학습 데이터 저장
                print(f"[💾 신규 학습 데이터] 새로운 패턴으로 저장 진행")
                learned_cmd = Chatbot(
                    user_email=user_email,
                    command=command,
                    intent=intent
                )
                learned_cmd.set_keywords_dict(keywords)
                
                db.session.add(learned_cmd)
                print(f"[✨ 새 학습 패턴 저장] 명령어: '{command}'")
                print(f"[🎯 저장된 Intent] {intent}")
                print(f"[🏷️ 저장된 키워드] {keywords}")
                print(f"[🚀 다음부터 고속 처리] 동일/유사 명령어는 0.05초 내 처리됩니다!")
            
            db.session.commit()
            print(f"[💾 DB 저장 완료] 학습 시스템이 더 똑똑해졌습니다!")
            
        except Exception as e:
            print(f"[❗ 학습 저장 오류] {str(e)}")
    
    def _extract_keywords_from_command(self, command):
        """명령어에서 6개 타입 키워드 추출"""
        keywords = {}
        command_lower = command.lower()
        
        # 7개 키워드 타입별 검사
        keyword_types = {
            'sender': ['교수님', '회사', '학과', 'naver', 'google', 'microsoft', '선생님', '교직원'],
            'date': ['오늘', '어제', '이번주', '지난주', '이번달', '지난달', '최근'],
            'tag': ['중요메일', '스팸', '보안경고', '회사', '대학교', '중요', '보안', '받은메일', '보낸메일', '받은', '보낸'],
            'attachment': ['첨부파일', '이미지', 'pdf', '문서', '파일', '사진', '동영상'],
            'action': ['검색', '보여줘', '찾아줘', '작성', '답장', '삭제', '요약', '써줘', '몇개', '개수', '통계', '개만', '개까지', '설정', '변경', '바꿔'],
            'content': ['과제', '회의', '공지', '영수증', '비밀번호', '로그인', '알림', '메일', '이메일', '프로젝트', '보고서'],
            'setting': ['다크모드', '라이트모드', '테마', '폰트', '크기', 'gmail', '페이지', '표시', '보내는', '이름']
        }
        
        for keyword_type, keyword_list in keyword_types.items():
            for keyword in keyword_list:
                if keyword in command_lower:
                    keywords[keyword_type] = keyword
                    break
        
        return keywords
    

import re
from sklearn.metrics.pairwise import cosine_similarity
from nomic import embed

class ChatbotService:
    def __init__(self, config, ai_models, email_service):
        self.config = config
        self.ai_models = ai_models
        self.email_service = email_service
        
        # 챗봇 의도 분류용 라벨
        self.candidate_labels = [
            "correct the vocabulary, spelling",
            "image generation using text", 
            "find something",
            "email search for a person"
        ]
        
        # 한국어 패턴 매칭
        self.korean_patterns = {
            "grammar": {
                "keywords": ["교정", "맞춤법", "문법", "틀렸", "고쳐", "수정"],
                "action": "grammar_correction"
            },
            "image": {
                "keywords": ["이미지", "그림", "사진", "그려", "만들어", "생성"],
                "action": "image_generation"
            },
            "person_search": {
                "keywords": ["님", "씨"],
                "required": ["메일", "이메일"],
                "action": "person_search"
            },
            "general_search": {
                "keywords": ["찾아", "검색", "찾기"],
                "action": "email_search"
            }
        }
    
    def process_user_input(self, user_input, user_email, app_password):
        """사용자 입력 처리"""
        try:
            print(f"[🤖 챗봇 요청] 사용자: {user_email}, 입력: {user_input}")
            
            if not user_input:
                return {"error": "입력이 비어있습니다."}, 400
            
            # 의도 분석
            intent_result = self._analyze_intent(user_input)
            
            print(f"[🎯 의도 분석] {intent_result['action']} (신뢰도: {intent_result['confidence']:.3f})")
            
            # 기능별 실행
            if intent_result['action'] == "grammar_correction":
                response = self._handle_grammar_correction(user_input)
            elif intent_result['action'] == "image_generation":
                response = self._handle_image_generation(user_input)
            elif intent_result['action'] == "email_search":
                response = self._handle_general_search(user_input, user_email, app_password)
            elif intent_result['action'] == "person_search":
                response = self._handle_person_search(user_input, user_email, app_password)
            else:
                response = self._handle_unknown_intent()
            
            return {
                "response": response,
                "action": intent_result['action'],
                "confidence": float(intent_result['confidence']),
                "detected_intent": intent_result['action'],
                "detection_method": intent_result['method']
            }, 200
            
        except Exception as e:
            print(f"[❗챗봇 오류] {str(e)}")
            return {"error": str(e)}, 500
    
    def _analyze_intent(self, user_input):
        """의도 분석 (영어 embedding + 한국어 키워드)"""
        # 1. 영어 Embedding 기반 분류
        try:
            text_inputs = [user_input] + self.candidate_labels
            result = embed.text(text_inputs, model='nomic-embed-text-v1', task_type='classification')
            
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
        
        # 2. 한국어 키워드 기반 분류
        korean_result = self._analyze_korean_patterns(user_input)
        
        # 3. 최종 의도 결정
        embedding_action_map = {
            "correct the vocabulary, spelling": "grammar_correction",
            "image generation using text": "image_generation", 
            "find something": "email_search",
            "email search for a person": "person_search"
        }
        
        embedding_action = embedding_action_map.get(embedding_label, "unknown")
        embedding_threshold = 0.25
        
        # 최종 결정
        if korean_result["confidence"] >= 0.3 and korean_result["confidence"] > embedding_score:
            return {
                'action': korean_result["action"],
                'confidence': korean_result["confidence"],
                'method': 'korean_keywords'
            }
        elif embedding_score >= embedding_threshold:
            return {
                'action': embedding_action,
                'confidence': embedding_score,
                'method': 'english_embedding'
            }
        else:
            return {
                'action': 'unknown',
                'confidence': max(korean_result["confidence"], embedding_score),
                'method': 'low_confidence'
            }
    
    def _analyze_korean_patterns(self, user_input):
        """한국어 패턴 분석"""
        user_input_lower = user_input.lower()
        
        korean_result = {"action": None, "confidence": 0.0, "matched_keywords": []}
        
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
                
                if confidence > korean_result["confidence"]:
                    korean_result = {
                        "action": pattern_info["action"],
                        "confidence": confidence,
                        "matched_keywords": matched_keywords
                    }
        
        return korean_result
    
    def _handle_grammar_correction(self, user_input):
        """문법 교정 처리"""
        try:
            # 교정할 텍스트 추출
            correction_text = user_input
            remove_words = ["교정해주세요", "교정해줘", "맞춤법", "문법", "correct", "spelling", "check", "fix"]
            for word in remove_words:
                correction_text = correction_text.replace(word, "").strip()
            
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
    
    def _handle_image_generation(self, user_input):
        """이미지 생성 처리 (비활성화됨)"""
        return """🎨 **이미지 생성 기능이 비활성화되었습니다**

죄송합니다. 현재 이미지 생성 기능은 사용할 수 없습니다.

🔧 **사용 가능한 기능들:**
• **문법/맞춤법 교정**: "이 문장 교정해주세요"
• **메일 검색**: "회의 관련 메일 찾아줘"  
• **사람별 메일**: "김철수님 메일 검색"

다른 기능을 사용해보세요! 😊"""
    
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
        """일반 이메일 검색"""
        try:
            # 검색 키워드 추출
            search_keywords = user_input.lower()
            remove_words = ["찾아줘", "찾아주세요", "검색해줘", "검색", "find", "search", "메일", "이메일", "email"]
            for word in remove_words:
                search_keywords = search_keywords.replace(word, "").strip()
            
            if not search_keywords:
                return "🔍 **메일 검색**\n\n검색하고 싶은 키워드를 입력해주세요.\n\n예시:\n• '회의 관련 메일 찾아줘'\n• '프로젝트 업데이트 검색'"
            
            # ✅ DB에서 이메일 검색 실행  
            try:
                found_emails = self._search_emails_in_db(user_email, search_keywords, max_results=50)
                
                if found_emails:
                    result = f"🔍 **검색 결과**\n\n키워드: '{search_keywords}'\n📧 찾은 메일: **{len(found_emails)}개**\n\n"
                    
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
            # Qwen으로 사람 이름/이메일 추출
            search_target = self._extract_search_target_with_qwen(user_input)
            
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
                # ✅ DB에서 사람별 이메일 검색 실행
                found_emails = self._search_emails_in_db(user_email, search_target, max_results=100)
                
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
    
    def _extract_search_target_with_qwen(self, text):
        """Qwen을 이용하여 검색 대상 추출"""
        # Qwen 모델이 로딩되지 않았다면 로딩 시도
        if not self.ai_models.load_qwen_model():
            print("[⚠️ Qwen 모델 없음 - 간단 추출 사용]")
            words = text.split()
            return " ".join(words[-2:]) if len(words) >= 2 else text
        
        try:
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

    def generate_ai_reply(self, sender, subject, body, current_user_email):
        """AI 답장 생성 (Qwen 1.5-1.8B 로컬 모델 사용)"""
        try:
            print(f"[🤖 AI 답장 요청] User: {current_user_email}, From: {sender}")
            
            # Qwen 로컬 모델 로딩 확인
            if not self.ai_models.load_qwen_model():
                return {'error': 'Qwen 모델을 로드할 수 없습니다.'}, 500
            
            # 프롬프트 생성
            user_prompt = self._build_ai_reply_prompt_for_qwen(sender, subject, body)
            
            inputs = self.ai_models.qwen_tokenizer(user_prompt, return_tensors="pt").to(self.ai_models.qwen_model.device)
            
            import torch
            with torch.no_grad():
                outputs = self.ai_models.qwen_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id,
                    pad_token_id=self.ai_models.qwen_tokenizer.pad_token_id
                )
            
            # 입력 부분 제거하고 생성된 답장만 추출
            generated_text = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # "assistant" 이후 텍스트만 가져오기
            if "assistant" in generated_text:
                ai_reply = generated_text.split("assistant")[-1].strip()
            else:
                ai_reply = generated_text[len(user_prompt):].strip()
            
            # 불필요한 부분 정리
            ai_reply = ai_reply.strip()
            if ai_reply.startswith('"') and ai_reply.endswith('"'):
                ai_reply = ai_reply[1:-1]
            
            print(f"[✅ AI 답장 생성 완료] User: {current_user_email}, 길이: {len(ai_reply)}자")
            
            return {'success': True, 'ai_reply': ai_reply}, 200
            
        except Exception as e:
            print(f"[❗AI 답장 생성 실패] {str(e)}")
            return {'error': f'AI 답장 생성 실패: {str(e)}'}, 500
    
    def _build_ai_reply_prompt_for_qwen(self, sender, subject, body):
        """Qwen 모델용 AI 답장 프롬프트 생성"""
        return f"""<|im_start|>system
You are a helpful email assistant that writes professional email replies.
<|im_end|>
<|im_start|>user
Please read the following email and write a polite, professional reply in English:

---
From: {sender}
Subject: {subject}
Body: {body}
---

Instructions:
1. Identify the purpose of the email (invitation, question, information request, scheduling, etc.)
2. Write a concise (3-4 sentences), polite reply that directly addresses the purpose
3. Use a friendly yet professional tone
4. Only output the reply text (no analysis, no quotes, no original email content)

Reply:
<|im_end|>
<|im_start|>assistant
"""

    def _search_emails_in_db(self, user_email, search_keywords, max_results=50):
        """DB에서 이메일 검색"""
        try:
            from models.tables import Mail
            from models.db import db
            import re
            
            # 이메일 주소 패턴 확인
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_found = re.search(email_pattern, search_keywords)
            
            if email_found:
                # 이메일 주소로 검색 (발신자 기준)
                search_email = email_found.group()
                print(f"[🎯 DB 이메일 주소 검색] {search_email}")
                
                db_results = Mail.query.filter(
                    Mail.user_email == user_email,
                    Mail.from_.contains(search_email)
                ).order_by(Mail.date.desc()).limit(max_results).all()
                
            else:
                # 키워드로 제목/내용/발신자 검색
                print(f"[🎯 DB 키워드 검색] {search_keywords}")
                
                db_results = Mail.query.filter(
                    Mail.user_email == user_email,
                    db.or_(
                        Mail.subject.contains(search_keywords),
                        Mail.body.contains(search_keywords),
                        Mail.from_.contains(search_keywords),
                        Mail.summary.contains(search_keywords)
                    )
                ).order_by(Mail.date.desc()).limit(max_results).all()
            
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
    
    def _build_ai_reply_prompt(self, sender, subject, body):
        """AI 답장을 위한 프롬프트 생성"""
        return f"""
You are a helpful email assistant that writes professional email replies.

Please read the following email and write a polite, professional reply in English:

---
From: {sender}
Subject: {subject}
Body: {body}
---

Instructions:
1. Identify the purpose of the email (invitation, question, information request, scheduling, etc.)
2. Write a concise (3-4 sentences), polite reply that directly addresses the purpose
3. Use a friendly yet professional tone
4. Only output the reply text (no analysis, no quotes, no original email content)

Reply:
""".strip()
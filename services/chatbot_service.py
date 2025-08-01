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
            
            # HuggingFace API 사용
            if not self.config.HF_TOKEN:
                return f"📝 **문법 교정 결과**\n\n원본: {correction_text}\n\n⚠️ HF_TOKEN이 설정되지 않아 교정 서비스를 사용할 수 없습니다."
            
            try:
                client = self.ai_models.get_inference_client()
                
                prompt = f"""다음 텍스트의 맞춤법, 문법, 띄어쓰기를 교정해주세요.

원본 텍스트:
"{correction_text}"

교정 지침:
1. 맞춤법 오류 수정
2. 문법 오류 수정  
3. 띄어쓰기 수정
4. 자연스러운 표현으로 개선
5. 원래 의미는 유지

교정된 텍스트:"""
                
                messages = [
                    {"role": "system", "content": "당신은 전문 교정 편집자입니다."},
                    {"role": "user", "content": prompt}
                ]
                
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=300,
                    temperature=0.3
                )
                
                corrected_text = response.choices[0].message.content.strip()
                
                return f"""📝 **문법 및 맞춤법 교정 완료**

**원본:**
{correction_text}

**교정된 텍스트:**
{corrected_text}

✅ **AI 교정이 완료되었습니다!**"""
                
            except Exception as e:
                # 간단한 규칙 기반 교정으로 fallback
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
        """이미지 생성 처리"""
        try:
            # 프롬프트 추출
            image_prompt = user_input
            remove_words = ["이미지 생성해주세요", "이미지 생성", "그려줘", "그림", "image generation", "generate", "만들어"]
            for word in remove_words:
                image_prompt = image_prompt.replace(word, "").strip()
            
            if not image_prompt:
                return "🎨 **이미지 생성**\n\n생성하고 싶은 이미지에 대한 설명을 입력해주세요.\n\n예시:\n• '아름다운 석양과 바다'\n• '귀여운 고양이가 놀고 있는 모습'"
            
            if not self.config.HF_TOKEN:
                return f"🎨 **이미지 생성**\n\n요청된 이미지: '{image_prompt}'\n\n⚠️ HF_TOKEN이 설정되지 않아 이미지 생성을 할 수 없습니다."
            
            try:
                from huggingface_hub import InferenceClient
                import base64
                import time
                import os
                
                client = InferenceClient(
                    model="runwayml/stable-diffusion-v1-5",
                    token=self.config.HF_TOKEN
                )
                
                # 한국어를 영어로 번역
                enhanced_prompt = self._translate_korean_to_english(image_prompt)
                enhanced_prompt = f"{enhanced_prompt}, high quality, detailed, beautiful, artistic"
                
                # 이미지 생성
                image_bytes = client.text_to_image(
                    prompt=enhanced_prompt,
                    height=512,
                    width=512,
                    num_inference_steps=20
                )
                
                # 파일 저장
                timestamp = int(time.time())
                filename = f"generated_image_{timestamp}.png"
                filepath = os.path.join(self.config.ATTACHMENT_FOLDER, filename)
                
                os.makedirs(self.config.ATTACHMENT_FOLDER, exist_ok=True)
                
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                
                return f"""🎨 **이미지 생성 완료!**

📝 **요청:** '{image_prompt}'
🖼️ **생성된 이미지:** {filename}
📁 **저장 위치:** /static/attachments/{filename}
🌐 **웹 주소:** http://localhost:5001/static/attachments/{filename}

✅ **성공!** 이미지가 생성되어 저장되었습니다."""
                
            except Exception as e:
                return f"🎨 **이미지 생성 실패**\n\n오류: {str(e)}\n\n💡 잠시 후 다시 시도해주세요."
                
        except Exception as e:
            return "❌ 이미지 생성 처리 중 오류가 발생했습니다."
    
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
            
            # 이메일 검색 실행
            try:
                found_emails = self.email_service.search_emails(user_email, app_password, search_keywords, max_results=50)
                
                if found_emails:
                    result = f"🔍 **검색 결과**\n\n키워드: '{search_keywords}'\n검색된 메일: {len(found_emails)}개\n\n"
                    for i, mail_info in enumerate(found_emails, 1):
                        result += f"**{i}. {mail_info['subject']}**\n"
                        result += f"📤 {mail_info['from']}\n"
                        result += f"📅 {mail_info['date']}\n"
                        if mail_info['preview']:
                            result += f"💬 {mail_info['preview']}\n"
                        result += "\n"
                    result += "💡 더 정확한 검색을 위해 구체적인 키워드를 사용해보세요."
                    return result
                else:
                    return f"🔍 **검색 결과**\n\n키워드: '{search_keywords}'\n\n❌ 관련된 메일을 찾을 수 없습니다.\n\n💡 다른 키워드로 시도해보세요."
                    
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
                # 사람별 이메일 검색 실행
                found_emails = self.email_service.search_emails(user_email, app_password, search_target, max_results=100)
                
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
                    result = f"👤 **사람별 메일 검색 결과**\n\n검색 대상: '{search_target}'\n발견된 메일: {len(person_emails)}개\n\n"
                    for i, mail_info in enumerate(person_emails, 1):
                        result += f"**{i}. {mail_info['subject']}**\n"
                        result += f"📤 {mail_info['from']}\n"
                        result += f"📅 {mail_info['date']}\n\n"
                    result += "💡 특정 메일을 자세히 보려면 메일 리스트에서 확인하세요."
                    return result
                else:
                    return f"👤 **사람별 메일 검색 결과**\n\n검색 대상: '{search_target}'\n\n❌ 해당 사람의 메일을 찾을 수 없습니다.\n\n💡 정확한 이름이나 이메일 주소로 다시 시도해보세요."
                    
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
        """AI 답장 생성"""
        try:
            print(f"[🤖 AI 답장 요청] User: {current_user_email}, From: {sender}")
            
            if not self.config.HF_TOKEN:
                return {'error': 'HF_TOKEN 환경 변수가 설정되어 있지 않습니다.'}, 500
            
            client = self.ai_models.get_inference_client()
            
            # 프롬프트 생성
            user_prompt = self._build_ai_reply_prompt(sender, subject, body)
            
            messages = [
                {"role": "system", "content": "You are a helpful email assistant that writes professional email replies."},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.7
            )
            
            ai_reply = response.choices[0].message.content.strip()
            
            print(f"[✅ AI 답장 생성 완료] User: {current_user_email}, 길이: {len(ai_reply)}자")
            
            return {'success': True, 'ai_reply': ai_reply}, 200
            
        except Exception as e:
            print(f"[❗AI 답장 생성 실패] {str(e)}")
            return {'error': f'AI 답장 생성 실패: {str(e)}'}, 500
    
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
from services.genie_qwen import genie_reply

class ReplyService:
    """AI 답장 생성 전용 서비스"""
    
    def __init__(self, ai_models):
        """ReplyService 초기화"""
        self.ai_models = ai_models
        print("[🤖 ReplyService 초기화 완료]")
    
    def generate_ai_reply(self, sender, subject, body, current_user_email, user_intent=""):
        """AI 답장 생성 (사용자 의도 반영, Qwen 1.5-1.8B 로컬 모델 사용)"""
        try:
            intent_log = f", 의도: {user_intent}" if user_intent else ""
            print(f"[🤖 AI 답장 요청] User: {current_user_email}, From: {sender}{intent_log}")
            
            # Qwen 로컬 모델 로딩 확인
            if not self.ai_models.load_qwen_model():
                return {'error': 'Qwen 모델을 로드할 수 없습니다.'}, 500
            
            # 이전 메일 기록에서 톤 분석
            tone_analysis = self._analyze_previous_email_tone(current_user_email, sender)
            
            # 프롬프트 생성 (사용자 의도 + 톤 분석 포함)
            user_prompt = self._build_ai_reply_prompt_for_qwen(sender, subject, body, user_intent, tone_analysis)

            try:
                generated_text = genie_reply(user_prompt)
                print(f"[프롬포트]{generated_text}")

                # 프롬프트 부분 제거하고 답장만 추출
                if "<|im_start|>assistant" in generated_text:
                    ai_reply = generated_text.split("<|im_start|>assistant")[-1].strip()
                elif "assistant" in generated_text:
                    ai_reply = generated_text.split("assistant")[-1].strip()
                else:
                    # 프롬프트 길이만큼 제거
                    ai_reply = generated_text[len(user_prompt):].strip()
                    # 만약 여전히 빈 문자열이면 전체 텍스트 사용
                    if not ai_reply:
                        ai_reply = generated_text.strip()
                
                # 불필요한 부분 정리
                ai_reply = ai_reply.strip()
                if ai_reply.startswith('"') and ai_reply.endswith('"'):
                    ai_reply = ai_reply[1:-1]

                print(f"[프롬포트]{ai_reply}")
                
                print(f"[✅ NPU AI 답장 생성 완료] User: {current_user_email}, 길이: {len(ai_reply)}자")

                return {'success': True, 'ai_reply': ai_reply}, 200

            except Exception as ge:
                print(f"[⚠️ NPU 답장 실패, Hf답장 시작] {ge}")

            try:
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

            except Exception as ex:
                print(f"[❗AI 답장 생성 실패] {str(ex)}")
                return {'error': f'AI 답장 생성 실패: {str(ex)}'}, 500

        except Exception as e:
            print(f"[❗AI 답장 생성 실패] {str(e)}")
            return {'error': f'AI 답장 생성 실패: {str(e)}'}, 500
    
    def _analyze_previous_email_tone(self, current_user_email, sender_email):
        """이전 메일 기록에서 톤 분석"""
        try:
            from models.tables import Mail
            from models.db import db
            
            # 발신자와 수신자 간의 이전 메일 기록 조회
            previous_mails = Mail.query.filter(
                db.and_(
                    Mail.user_email == current_user_email,
                    Mail.from_.contains(sender_email)
                )
            ).order_by(Mail.date.desc()).limit(5).all()
            
            if not previous_mails:
                return "formal"
            
            # 이전 메일 샘플 텍스트 수집
            sample_texts = []
            for mail in previous_mails:
                if mail.body and len(mail.body.strip()) > 20:
                    clean_body = mail.body.replace('\n', ' ').strip()[:200]
                    sample_texts.append(clean_body)
            
            if not sample_texts:
                return "formal"
            
            # Qwen으로 톤 분석
            analysis_prompt = f"""\
<|im_start|>system
"당신은 이메일 톤 분석 전문가입니다. "
"주어진 이전 메일 교환을 보고 적절한 답장 톤을 한 단어로만 추천하세요. "
"가능한 값: formal, casual, professional. 다른 말/기호/번역 금지."
<|im_end|>
<|im_start|>user
이전 메일 교환 내역을 분석하여 적절한 답장 톤을 추천해주세요.

이전 메일 샘플:
{chr(10).join(sample_texts)}

다음 중 하나의 톤을 선택해주세요:
- formal: 정중하고 공식적인 톤
- casual: 친근하고 편안한 톤
- professional: 전문적이고 비즈니스적인 톤

답변은 톤 이름만 응답하세요.
<|im_end|>
<|im_start|>assistant
"""
            try:
                # 1) NPU(Genie) 경로
                tone_result = genie_reply(analysis_prompt)
                print(f"[✅ Genie 사용] : NPU를 활용한 톤추출 - {tone_result}")
                # 톤 추출
                if "formal" in tone_result.lower():
                    return "formal"
                elif "casual" in tone_result.lower():
                    return "casual"
                elif "professional" in tone_result.lower():
                    return "professional"
                else:
                    return "formal"
                
            except Exception as ge:
                print(f"[⚠️ Genie 톤 분석 실패, Hf분석 시작] {ge}")
            
            try:
                inputs = self.ai_models.qwen_tokenizer(analysis_prompt, return_tensors="pt").to(self.ai_models.qwen_model.device)
            
                import torch
                with torch.no_grad():
                  outputs = self.ai_models.qwen_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.3,
                    do_sample=True,
                    eos_token_id=self.ai_models.qwen_tokenizer.eos_token_id
                  )
            
                tone_result = self.ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
                # 톤 추출
                if "formal" in tone_result.lower():
                    return "formal"
                elif "casual" in tone_result.lower():
                    return "casual"
                elif "professional" in tone_result.lower():
                    return "professional"
                else:
                    return "formal"

            except Exception as ex:
                print(f"[⚠️ 톤 분석 오류] {str(ex)}")
                return "formal"

        except Exception as e:
            print(f"[⚠️ 톤 분석 오류] {str(e)}")
            return "formal"
    
    def _build_ai_reply_prompt_for_qwen(self, sender, subject, body, user_intent="", tone_analysis=None):
        """Qwen 모델용 AI 답장 프롬프트 생성 (사용자 의도 + 톤 분석 반영)"""
        
        # 사용자 의도에서 언어와 내용 분석
        language_instruction = "in English"
        content_guidance = ""
        
        if user_intent:
            user_intent_lower = user_intent.lower()
            
            # 언어 감지
            korean_keywords = ['한국어', '한글', '한국말', '너무', '정말', '그리고', '하지만', '때문에']
            if any(keyword in user_intent_lower for keyword in korean_keywords) or any(ord(char) > 127 for char in user_intent):
                language_instruction = "in Korean"
            
            # 내용 안내
            if "거절" in user_intent_lower or "decline" in user_intent_lower:
                content_guidance = "You should politely decline or say no to the request."
            elif "수락" in user_intent_lower or "accept" in user_intent_lower:
                content_guidance = "You should accept the request positively."
            elif "질문" in user_intent_lower or "문의" in user_intent_lower or "question" in user_intent_lower:
                content_guidance = "You should ask relevant questions or request more information."
            elif "감사" in user_intent_lower or "thank" in user_intent_lower:
                content_guidance = "You should express gratitude and appreciation."
        
        # 톤 설정
        tone_instruction = ""
        if tone_analysis == "casual":
            tone_instruction = "Use a friendly and casual tone."
        elif tone_analysis == "professional":
            tone_instruction = "Use a professional and business-like tone."
        else:
            tone_instruction = "Use a polite and formal tone."
        
        # 프롬프트 구성
        prompt = f"""<|im_start|>system
You are a professional email assistant. Write a proper email reply {language_instruction}.

Guidelines:
- {tone_instruction}
- Be concise and clear
- Address the main points from the original email
- {content_guidance}
- Do not include subject line or email headers
- Write only the email body content
<|im_end|>

<|im_start|>user
Original Email:
From: {sender}
Subject: {subject}
Message: {body}

User Intent: {user_intent if user_intent else 'Write an appropriate reply'}

Please write a reply email body:
<|im_end|>

<|im_start|>assistant
"""
        
        return prompt
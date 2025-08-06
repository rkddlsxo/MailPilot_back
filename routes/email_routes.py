from flask import Blueprint, request, jsonify
from datetime import datetime

def create_email_routes(email_service, ai_models, session_manager, attachment_service):
    email_bp = Blueprint('email', __name__)


    @email_bp.route('/api/emails/stored', methods=['POST'])
    def get_stored_emails():
        """DB에서 저장된 이메일 조회"""
        try:
            data = request.get_json()
            email = data.get("email")

            if not session_manager.session_exists(email):
                return jsonify({"error": "로그인이 필요합니다."}), 401

            from models.tables import Mail  # ✅ 필요시 상단으로 이동

            mails = Mail.query.filter_by(user_email=email)\
                            .order_by(Mail.date.desc())\
                            .limit(20).all()

            result = [{
                "id": mail.mail_id,
                "subject": mail.subject,
                "from": mail.from_,
                "date": mail.date.strftime('%Y-%m-%d %H:%M:%S'),
                "body": mail.body[:1000]
            } for mail in mails]

            return jsonify({
                "emails": result,
                "source": "database",
                "count": len(result)
            })

        except Exception as e:
            print(f"[❗DB 메일 조회 오류] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @email_bp.route('/api/summary', methods=['POST'])
    def get_email_summary():
        """이메일 목록 가져오기 (첨부파일 처리 포함)"""
        try:
            data = request.get_json()
            username = data.get("email")
            app_password = data.get("app_password")
            
            # 사용자 세션 확인
            if not session_manager.session_exists(username):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            print(f"[📧 메일 요청] 사용자: {username}")
            
            # 날짜 필터링 처리
            after_date = data.get("after")
            after_dt = None
            if after_date:
                try:
                    after_date_clean = after_date.replace("Z", "+00:00")
                    after_dt = datetime.fromisoformat(after_date_clean)
                    after_dt = after_dt.replace(tzinfo=None)
                    print(f"[📅 필터링 기준] {after_dt} 이후 메일만 가져옴")
                except Exception as e:
                    print("[⚠️ after_date 파싱 실패]", e)
            
            # 메일 수 결정
            count = 10 if after_dt else 5
            
            # 이메일 가져오기
            raw_emails = email_service.fetch_emails(username, app_password, count, after_dt)
            
            # 이메일 처리 (분류, 요약, 첨부파일)
            processed_emails = []
            for email_data in raw_emails:
                try:
                    # AI 분류
                    classification_result = ai_models.classify_email(email_data['body'])
                    
                    # 요약 생성
                    if ai_models.load_summarizer():
                        try:
                            if not email_data['body']:
                                summary_text = "(본문 없음)"
                            else:
                                safe_text = email_data['body'][:1000]
                                if len(safe_text) < 50:
                                    summary_text = safe_text
                                else:
                                    summary_text = ai_models.summarizer(
                                        safe_text,
                                        max_length=80,
                                        min_length=30,
                                        do_sample=False
                                    )[0]["summary_text"]
                        except Exception as e:
                            print("[⚠️ 요약 실패]", str(e))
                            summary_text = email_data['body'][:150] + "..." if email_data['body'] else "(요약 실패)"
                    else:
                        summary_text = email_data['body'][:150] + "..." if email_data['body'] else "(요약 없음)"
                    
                    # 첨부파일 처리
                    attachments = []
                    if email_data.get('raw_message'):
                        attachments = attachment_service.process_email_attachments(
                            email_data['raw_message'], 
                            email_data['subject'], 
                            str(email_data['id'])
                        )
                    
                    # 태그 결정 (간단한 방식)
                    tag = "받은"
                    if "important" in classification_result['classification'].lower():
                        tag = "중요"
                    elif "spam" in classification_result['classification'].lower():
                        tag = "스팸"
                    
                    processed_email = {
                        "id": email_data["id"],
                        "subject": email_data["subject"],
                        "from": email_data["from"],
                        "date": email_data["date"],
                        "body": email_data["body"][:1000],  # 본문 제한
                        "tag": tag,
                        "summary": summary_text,
                        "classification": classification_result['classification'],
                        "attachments": attachments,
                        "has_attachments": len(attachments) > 0,
                        "attachment_summary": attachment_service.generate_attachment_summary(attachments) if attachments else ""
                    }
                    
                    processed_emails.append(processed_email)
                    
                except Exception as e:
                    print(f"[⚠️ 이메일 처리 오류] {str(e)}")
                    # 기본 처리된 이메일이라도 포함
                    processed_emails.append({
                        "id": email_data["id"],
                        "subject": email_data["subject"],
                        "from": email_data["from"],
                        "date": email_data["date"],
                        "body": email_data["body"][:1000],
                        "tag": "받은",
                        "summary": email_data["body"][:150] + "..." if email_data["body"] else "(처리 실패)",
                        "classification": "unknown",
                        "attachments": [],
                        "has_attachments": False,
                        "attachment_summary": ""
                    })
            
            # 최신순 정렬
            processed_emails.sort(key=lambda x: x['date'], reverse=True)
            
            # 세션에 저장
            user_session = session_manager.get_session(username)
            if user_session:
                user_session['last_emails'] = processed_emails
                user_session['last_update'] = datetime.now().isoformat()
            
            print(f"[📊 결과] 사용자: {username}, 총 {len(processed_emails)}개 메일 처리 완료")
            
            return jsonify({
                "emails": processed_emails,
                "user_session": session_manager.get_user_key(username)[:8] + "...",
                "cache_info": f"세션에 {len(processed_emails)}개 메일 저장됨"
            })
            
        except Exception as e:
            print("[❗에러 발생]", str(e))
            return jsonify({"error": str(e)}), 500
    
    @email_bp.route('/api/send', methods=['POST'])
    def send_email():
        """이메일 발송"""
        try:
            data = request.get_json()
            sender_email = data["email"]
            app_password = data["app_password"]
            to = data["to"]
            subject = data["subject"]
            body = data["body"]
            
            # 사용자 세션 확인
            if not session_manager.session_exists(sender_email):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            # 이메일 발송
            success = email_service.send_email(sender_email, app_password, to, subject, body)
            
            if success:
                return jsonify({"message": "✅ 메일 전송 성공"}), 200
            else:
                return jsonify({"error": "메일 전송 실패"}), 500
                
        except Exception as e:
            print("[❗메일 전송 실패]", str(e))
            return jsonify({"error": str(e)}), 500
    
    @email_bp.route('/api/email-search', methods=['POST'])
    def search_emails():
        """이메일 검색"""
        try:
            data = request.get_json()
            user_input = data.get("user_input", "").strip()
            user_email = data.get("email", "")
            app_password = data.get("app_password", "")
            
            if not all([user_input, user_email, app_password]):
                return jsonify({"error": "사용자 입력, 이메일, 앱 비밀번호가 모두 필요합니다."}), 400
                
            # 사용자 세션 확인
            if not session_manager.session_exists(user_email):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            print(f"[🔍 이메일 검색 요청] 사용자: {user_email}, 입력: {user_input}")
            
            # 검색 대상 추출 (Qwen 사용)
            if ai_models.load_qwen_model():
                try:
                    search_target = extract_search_target_with_qwen(user_input, ai_models)
                    print(f"[🎯 검색 대상 추출] '{search_target}'")
                except Exception as e:
                    print(f"[⚠️ Qwen 추출 실패] {str(e)}")
                    # 간단한 이메일 추출 fallback
                    import re
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    emails_found = re.findall(email_pattern, user_input)
                    if emails_found:
                        search_target = emails_found[0]
                    else:
                        words = user_input.split()
                        search_target = " ".join(words[-2:]) if len(words) >= 2 else user_input
            else:
                words = user_input.split()
                search_target = " ".join(words[-2:]) if len(words) >= 2 else user_input
            
            # 이메일 검색 실행
            found_emails = email_service.search_emails(user_email, app_password, search_target, max_results=100)
            
            return jsonify({
                "success": True,
                "search_target": search_target,
                "results": found_emails,
                "found_count": len(found_emails),
                "confidence": 1.0,
                "detected_intent": "email_search_completed"
            })
            
        except Exception as e:
            print(f"[❗이메일 검색 오류] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return email_bp

def extract_search_target_with_qwen(text, ai_models):
    """Qwen을 이용하여 검색 대상 추출"""
    try:
        prompt = (
            "<|im_start|>system\nYou are an email assistant. "
            "Your job is to extract the email address or name the user is referring to. "
            "You must always respond in the format: The user is referring to ... \n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        inputs = ai_models.qwen_tokenizer(prompt, return_tensors="pt").to(ai_models.qwen_model.device)
        
        import torch
        with torch.no_grad():
            outputs = ai_models.qwen_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=ai_models.qwen_tokenizer.eos_token_id
            )
        
        decoded_output = ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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
        words = text.split()
        return " ".join(words[-2:]) if len(words) >= 2 else text




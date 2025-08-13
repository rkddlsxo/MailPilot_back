from flask import Blueprint, request, jsonify
from datetime import datetime
import json
from models.tables import db, Mail, Todo

def create_email_routes(email_service, ai_models, session_manager, attachment_service, todo_service):
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
                "body": mail.body[:1000],
                "tag": mail.tag or "받은",
                "summary": mail.summary or "요약 없음",
                "classification": mail.classification or "unknown",
                "attachments": json.loads(mail.attachments_data).get('files', []) if mail.attachments_data else [],
                "has_attachments": json.loads(mail.attachments_data).get('has_attachments', False) if mail.attachments_data else False,
                "attachment_summary": json.loads(mail.attachments_data).get('summary', '') if mail.attachments_data else ""
            } for mail in mails]

            return jsonify({
                "emails": result,
                "source": "database",
                "count": len(result)
            })

        except Exception as e:
            print(f"[❗DB 메일 조회 오류] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @email_bp.route('/api/emails/sent', methods=['POST'])
    def get_sent_emails():
        """보낸메일 가져오기 (AI 처리 없음)"""
        try:
            data = request.get_json()
            email = data.get("email")
            count = data.get("count", 5)
            app_password = data.get("app_password")  # 앱 비밀번호 직접 받기
            
            if not session_manager.session_exists(email):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            # DB에서 기존 보낸메일 확인 (중복 체크용)
            from models.tables import Mail
            existing_sent_mails_dict = {}
            existing_sent_mails = Mail.query.filter_by(
                user_email=email, 
                mail_type='sent'
            ).all()
            
            # 기존 메일을 딕셔너리로 변환 (mail_id를 키로)
            for mail in existing_sent_mails:
                existing_sent_mails_dict[mail.mail_id] = {
                    "id": mail.mail_id,
                    "subject": mail.subject,
                    "from": mail.from_,
                    "date": mail.date.strftime('%Y-%m-%d %H:%M:%S'),
                    "body": mail.body[:1000],
                    "tag": "보낸",
                    "summary": "(보낸 메일)",
                    "classification": "sent",
                    "attachments": [],
                    "has_attachments": False,
                    "attachment_summary": ""
                }
            
            print(f"[📤 DB 기존 보낸메일] {len(existing_sent_mails_dict)}개 확인")
            
            # DB에 없으면 Gmail에서 가져오기
            print(f"[📤 Gmail 보낸메일] 가져오기 시작...")
            
            if not app_password:
                return jsonify({"error": "앱 비밀번호가 필요합니다."}), 400
            
            try:
                # 보낸메일 가져오기 (AI 처리 없음)
                sent_emails = email_service.fetch_sent_emails(
                    email,  # 이메일 직접 사용 
                    app_password,  # 앱 비밀번호 직접 사용
                    count=count
                )
                
                if sent_emails:
                    # Gmail 보낸메일과 DB 기존 메일 통합 처리
                    processed_emails = []
                    new_emails_saved = 0
                    
                    for email_data in sent_emails:
                        email_id = str(email_data['id'])
                        
                        # ✅ 기존 메일인지 확인
                        if email_id in existing_sent_mails_dict:
                            print(f"[🔄 기존 메일] {email_data['subject'][:30]}...")
                            processed_emails.append(existing_sent_mails_dict[email_id])
                        else:
                            # ✅ 새 메일만 DB에 저장
                            try:
                                new_mail = Mail(
                                    mail_id=email_id,
                                    user_email=email,
                                    subject=email_data["subject"],
                                    from_=email_data["from"],
                                    date=email_data["date_obj"],
                                    body=email_data["body"],
                                    tag="보낸",
                                    summary="(보낸 메일)",
                                    classification="sent",
                                    raw_message=email_data.get('raw_message', ''),
                                    attachments_data='{"summary":"","count":0,"has_attachments":false,"files":[]}',
                                    mail_type='sent'
                                )
                                db.session.add(new_mail)
                                db.session.commit()  # 즉시 커밋으로 중복 방지
                                print(f"[💾 새 보낸메일 저장] {email_data['subject'][:30]}...")
                                new_emails_saved += 1
                            except Exception as save_error:
                                db.session.rollback()
                                if "Duplicate entry" in str(save_error):
                                    print(f"[🔄 중복 감지 건너뛰기] {email_data['subject'][:30]}...")
                                else:
                                    print(f"[❗ 저장 오류] {email_data['subject'][:30]}...: {str(save_error)}")
                            
                            # 응답용 데이터 추가
                            processed_emails.append({
                                "id": email_data["id"],
                                "subject": email_data["subject"],
                                "from": email_data["from"],
                                "date": email_data["date"],
                                "body": email_data["body"][:1000],
                                "tag": "보낸",
                                "summary": "(보낸 메일)",
                                "classification": "sent",
                                "attachments": [],
                                "has_attachments": False,
                                "attachment_summary": ""
                            })
                    
                    print(f"[📤 보낸메일] 총 {len(processed_emails)}개 처리 완료 (신규 저장: {new_emails_saved}개)")
                    
                    return jsonify({
                        "emails": processed_emails,
                        "source": "gmail_sent_integrated",
                        "count": len(processed_emails)
                    })
                else:
                    return jsonify({
                        "emails": [],
                        "source": "gmail_sent",
                        "count": 0,
                        "message": "보낸메일이 없습니다."
                    })
                    
            except Exception as e:
                print(f"[❗보낸메일 가져오기 오류] {str(e)}")
                return jsonify({"error": f"보낸메일 가져오기 실패: {str(e)}"}), 500
                
        except Exception as e:
            print(f"[❗보낸메일 API 오류] {str(e)}")
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
            
            # 1. Gmail에서 원본 메일 가져오기 (AI 처리 없음)
            print(f"[🔍 디버그] Gmail 연결 시도 중...")
            try:
                raw_emails = email_service.fetch_emails(username, app_password, count, after_dt)
                print(f"[📥 Gmail] {len(raw_emails)}개 원본 메일 가져옴")
            except Exception as gmail_error:
                print(f"[❗ Gmail 연결 실패] {str(gmail_error)}")
                return jsonify({"error": f"Gmail 연결 실패: {str(gmail_error)}"}), 500
            
            # 2. DB에서 기존 메일들 조회
            print(f"[🔍 디버그] DB에서 기존 메일 조회 중...")
            from models.tables import Mail
            existing_mails = {}
            try:
                # DB 세션 상태 확인 및 복구
                try:
                    db.session.rollback()  # 이전 에러로 인한 세션 정리
                except:
                    pass
                    
                db_mails = Mail.query.filter_by(user_email=username).all()
                existing_mails = {mail.mail_id: {
                    "id": mail.mail_id,
                    "subject": mail.subject,
                    "from": mail.from_,
                    "date": mail.date.strftime('%Y-%m-%d %H:%M:%S'),
                    "body": mail.body[:1000],
                    "tag": mail.tag or "받은",
                    "summary": mail.summary or "요약 없음",
                    "classification": mail.classification or "unknown",
                    "raw_message": mail.raw_message,
                    "attachments": json.loads(mail.attachments_data).get('files', []) if mail.attachments_data else [],
                    "has_attachments": json.loads(mail.attachments_data).get('has_attachments', False) if mail.attachments_data else False,
                    "attachment_summary": json.loads(mail.attachments_data).get('summary', '') if mail.attachments_data else ""
                } for mail in db_mails}
                print(f"[💾 DB] {len(existing_mails)}개 기존 메일 확인")
            except Exception as e:
                print(f"[⚠️ DB 조회 실패] {str(e)}")
                existing_mails = {}
            
            # 3. ✅ 통합 처리: Gmail 메일별로 DB 확인 후 AI 처리 결정
            processed_emails = []
            new_emails_processed = 0
            
            for email_data in raw_emails:
                try:
                    email_id = str(email_data['id'])
                    
                    # 🔍 디버깅: Gmail ID와 DB ID 비교 로그 추가
                    print(f"[🔍 ID 비교] Gmail 메일 ID: {email_id}, 제목: {email_data['subject'][:30]}...")
                    print(f"[🔍 기존 DB IDs] {list(existing_mails.keys())[:3]}...") # 첫 3개만 표시
                    
                    # ✅ DB에 있는 메일인지 확인
                    if email_id in existing_mails:
                        print(f"[♾️ DB 사용 - ID 매칭됨] {email_data['subject'][:30]}...")
                        processed_emails.append(existing_mails[email_id])
                        continue
                    
                    # ✅ 새 메일 → AI 처리 + DB 저장
                    mail_type = email_data.get('mail_type', 'inbox')
                    
                    if mail_type == 'sent':
                        print(f"[📤 보낸메일 - AI 처리 건너뛰기] {email_data['subject'][:30]}...")
                        # 보낸메일은 기본 정보만 저장
                        classification_result = {'classification': 'sent'}
                        attachments_json = {
                            "summary": "",
                            "count": 0,
                            "has_attachments": False,
                            "files": []
                        }
                        summary = ""
                        todos_json = {"todos": []}
                    else:
                        print(f"[🤖 AI 처리 시작] {email_data['subject'][:30]}...")
                        new_emails_processed += 1
                        
                        # ✅ AI 분류
                        print(f"[🔍 AI 분류] {email_data['subject'][:30]}...")
                        try:
                            classification_result = ai_models.classify_email(email_data['body'])
                            print(f"[✅ 분류 완료] {classification_result['classification']}")
                        except Exception as e:
                            print(f"[❗ AI 분류 오류] {str(e)}")
                            classification_result = {'classification': 'unknown'}
                    
                        # ✅ 첨부파일 처리 (받은메일만, AI 요약 생성 전에 먼저 처리)
                        print(f"[🔍 첨부파일] {email_data['subject'][:30]}...")
                        attachments_json = {
                            "summary": "",
                            "count": 0,
                            "has_attachments": False,
                            "files": []
                        }
                        
                        if email_data.get('raw_message'):
                            try:
                                attachments = attachment_service.process_email_attachments(
                                    email_data['raw_message'], 
                                    email_data['subject'], 
                                    str(email_data['id'])
                                )
                                
                                if attachments:
                                    attachments_json = {
                                        "summary": attachment_service.generate_attachment_summary(attachments),
                                        "count": len(attachments),
                                        "has_attachments": True,
                                        "files": attachments
                                    }
                                    print(f"[✅ 첨부파일] {len(attachments)}개 처리 완료")
                                else:
                                    print(f"[📎 첨부파일] 없음")
                            except Exception as e:
                                print(f"[❗ 첨부파일 오류] {str(e)}")
                        else:
                            print(f"[⚠️ raw_message 없음] {email_data['subject'][:30]}...")
                    
                        # ✅ AI 요약 생성 (받은메일만, OCR 텍스트 포함)
                        print(f"[🔍 AI 요약] {email_data['subject'][:30]}...")
                        summary = ""
                        try:
                            if not email_data['body']:
                                summary = "(본문 없음)"
                            else:
                                # OCR 텍스트와 이메일 본문 결합
                                full_content_for_summary = email_data['body']
                                
                                # 이미지 OCR 텍스트 추가
                                if attachments_json.get('files'):
                                    image_texts = []
                                    for attachment in attachments_json['files']:
                                        if (attachment.get('type') == 'image' and 
                                            attachment.get('extracted_text') and 
                                            attachment.get('ocr_success')):
                                            image_text = attachment['extracted_text'].strip()
                                            if image_text:
                                                image_texts.append(f"[이미지: {attachment['filename']}]\n{image_text}")
                                    
                                    if image_texts:
                                        full_content_for_summary += f"\n\n--- 첨부 이미지 텍스트 ---\n" + "\n\n".join(image_texts)
                                        print(f"[🖼️ 요약용 OCR 통합] {len(image_texts)}개 이미지 텍스트 포함")
                                
                                # OCR 포함된 내용으로 요약 생성
                                summary = _summarize_with_qwen(full_content_for_summary, ai_models)
                                print(f"[✅ 요약 완료] {summary[:50]}...")
                        except Exception as e:
                            print(f"[❗ AI 요약 오류] {str(e)}")
                            summary = "(요약 생성 실패)"
                        
                        # ✅ 할일 추출 (받은메일만) - OCR 텍스트 포함
                        print(f"[📋 할일 추출] {email_data['subject'][:30]}...")
                        try:
                            # OCR 텍스트와 이메일 본문 결합
                            full_content_for_todo = email_data['body']
                            
                            # 이미지 OCR 텍스트 추가
                            if attachments_json.get('files'):
                                image_texts = []
                                for attachment in attachments_json['files']:
                                    if (attachment.get('type') == 'image' and 
                                        attachment.get('extracted_text') and 
                                        attachment.get('ocr_success')):
                                        image_text = attachment['extracted_text'].strip()
                                        if image_text:
                                            image_texts.append(f"[이미지: {attachment['filename']}]\n{image_text}")
                                
                                if image_texts:
                                    full_content_for_todo += f"\n\n--- 첨부 이미지 텍스트 ---\n" + "\n\n".join(image_texts)
                                    print(f"[🖼️ 할일용 OCR 통합] {len(image_texts)}개 이미지 텍스트 포함")
                            
                            todo_result = todo_service.extract_todos_from_email(
                                email_body=full_content_for_todo,  # OCR 텍스트가 포함된 통합 내용
                                email_subject=email_data['subject'], 
                                email_from=email_data['from'],
                                email_date=email_data['date']
                            )
                            
                            if todo_result['success'] and todo_result['todos']:
                                # 기존 할일과 중복 체크
                                existing_todos = Todo.query.filter_by(user_email=username).all()
                                existing_keys = {f"{todo.title.lower().strip()}_{todo.type}" for todo in existing_todos}
                                
                                new_todos_count = 0
                                for todo in todo_result['todos']:
                                    todo_key = f"{todo['title'].lower().strip()}_{todo['type']}"
                                    
                                    if todo_key not in existing_keys:
                                        # 중복이 아니면 DB 저장
                                        todo_date = None
                                        if todo.get('date'):
                                            try:
                                                todo_date = datetime.strptime(todo['date'], '%Y-%m-%d').date()
                                            except:
                                                pass
                                        
                                        new_todo = Todo(
                                            user_email=username,
                                            title=todo['title'],
                                            type=todo['type'], 
                                            event=todo.get('description', ''),
                                            date=todo_date,
                                            time=todo.get('time'),
                                            priority=todo.get('priority', 'medium'),
                                            status='pending',
                                            mail_id=str(email_data['id'])
                                        )
                                        
                                        db.session.add(new_todo)
                                        existing_keys.add(todo_key)  # 메모리에서도 중복 방지
                                        new_todos_count += 1
                                
                                if new_todos_count > 0:
                                    db.session.commit()
                                    print(f"[✅ 할일 추출 완료] {new_todos_count}개 새 할일 생성")
                                else:
                                    print(f"[📋 할일 추출] 중복 없음, 새 할일 없음")
                            else:
                                print(f"[📋 할일 추출] 할일 없음")
                                
                        except Exception as e:
                            print(f"[❗ 할일 추출 오류] {str(e)}")
                            # 할일 추출 실패해도 이메일 처리는 계속 진행
                            todos_json = {"todos": []}
                    
                    # 태그 결정 (받은메일만)
                    if mail_type == 'inbox':
                        tag = "받은"
                        if "important" in classification_result['classification'].lower():
                            tag = "중요"
                        elif "spam" in classification_result['classification'].lower():
                            tag = "스팸"
                    else:
                        tag = "보낸"
                    
                    processed_email = {
                        "id": email_data["id"],
                        "subject": email_data["subject"],
                        "from": email_data["from"],
                        "date": email_data["date"],
                        "body": email_data["body"][:1000],  # 본문 제한
                        "tag": tag,
                        "summary": summary,
                        "classification": classification_result['classification'],
                        "attachments": attachments_json.get('files', []),
                        "has_attachments": attachments_json.get('has_attachments', False),
                        "attachment_summary": attachments_json.get('summary', '')
                    }
                    
                    processed_emails.append(processed_email)
                    
                    # ✅ DB에 새 메일 저장
                    print(f"[💾 DB 저장] {email_data['subject'][:30]}...")
                    try:
                        new_mail = Mail(
                            mail_id=email_id,
                            user_email=username,
                            subject=email_data["subject"],
                            from_=email_data["from"],
                            date=email_data["date_obj"],  # 날짜 객체 직접 사용
                            body=email_data["body"],
                            tag=tag,
                            summary=summary,
                            classification=classification_result['classification'],
                            raw_message=email_data.get('raw_message', ''),
                            attachments_data=json.dumps(attachments_json, ensure_ascii=False),
                            mail_type=mail_type  # 'inbox' 또는 'sent'
                        )
                        db.session.add(new_mail)
                        db.session.commit()
                        print(f"[✅ DB 저장 완료] {email_data['subject'][:30]}...")
                        
                    except Exception as db_error:
                        print(f"[⚠️ DB 저장 실패] {str(db_error)}")
                        try:
                            db.session.rollback()
                            print(f"[🔄 DB 세션 롤백] 계속 진행...")
                        except:
                            pass
                    
                except Exception as e:
                    print(f"[⚠️ 이메일 처리 오류] {str(e)}")
                    # DB에 있는지 확인 후 기본 처리
                    email_id = str(email_data['id'])
                    if email_id in existing_mails:
                        processed_emails.append(existing_mails[email_id])
                    else:
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
            
            # 세션 저장 제거 - 모든 데이터는 DB에 저장됨
            
            print(f"[📊 결과] 사용자: {username}, 총 {len(processed_emails)}개 메일 (신규 AI 처리: {new_emails_processed}개, DB 사용: {len(processed_emails)-new_emails_processed}개)")
            
            return jsonify({
                "emails": processed_emails,
                "user_session": session_manager.get_user_key(username)[:8] + "...",
                "cache_info": f"DB: {len(processed_emails)-new_emails_processed}개, 신규 처리: {new_emails_processed}개"
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
            
            print(f"[🔍 DB 이메일 검색] 사용자: {user_email}, 검색어: {user_input}")
            
            # ✅ DB에서 직접 검색 (Gmail IMAP 대신)
            from models.tables import Mail
            
            try:
                # 검색어 처리 (이메일 주소 또는 키워드)
                import re
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                email_found = re.search(email_pattern, user_input)
                
                if email_found:
                    # 이메일 주소로 검색
                    search_email = email_found.group()
                    print(f"[🎯 이메일 주소 검색] {search_email}")
                    
                    db_results = Mail.query.filter(
                        Mail.user_email == user_email,
                        Mail.from_.contains(search_email)
                    ).order_by(Mail.date.desc()).limit(50).all()
                    
                else:
                    # 키워드로 제목/내용 검색
                    print(f"[🎯 키워드 검색] {user_input}")
                    
                    db_results = Mail.query.filter(
                        Mail.user_email == user_email,
                        db.or_(
                            Mail.subject.contains(user_input),
                            Mail.body.contains(user_input),
                            Mail.from_.contains(user_input)
                        )
                    ).order_by(Mail.date.desc()).limit(50).all()
                
                # 결과 포맷팅
                found_emails = []
                for mail in db_results:
                    found_emails.append({
                        "id": mail.mail_id,
                        "subject": mail.subject[:60] + "..." if len(mail.subject) > 60 else mail.subject,
                        "from": mail.from_[:40] + "..." if len(mail.from_) > 40 else mail.from_,
                        "date": mail.date.strftime('%Y-%m-%d %H:%M:%S'),
                        "preview": mail.body[:200] + "..." if len(mail.body) > 200 else mail.body,
                        "classification": mail.classification,
                        "summary": mail.summary
                    })
                
                print(f"[✅ DB 검색 완료] {len(found_emails)}개 결과")
                
            except Exception as db_error:
                print(f"[❗ DB 검색 실패] {str(db_error)}")
                found_emails = []
            
            return jsonify({
                "success": True,
                "search_target": user_input,
                "results": found_emails,
                "found_count": len(found_emails),
                "confidence": 1.0,
                "detected_intent": "db_search_completed",
                "source": "database"
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

def _summarize_with_qwen(text, ai_models):
    """Qwen 1.5-1.8B 모델로 이메일 본문 요약"""
    try:
        # AI 모델이 있는지 확인 후 사용
        if ai_models and ai_models.load_qwen_model():
            try:
                # 텍스트 길이 제한
                safe_text = text[:800]
                
                prompt = f"""<|im_start|>system
당신은 이메일 요약 전문가입니다.
<|im_end|>
<|im_start|>user
다음 이메일 본문을 간단히 요약해주세요:

{safe_text}

요약 지침:
1. 핵심 내용만 2-3문장으로 요약
2. 80자 이내로 간결하게
3. 한국어로 응답

요약:
<|im_end|>
<|im_start|>assistant
"""
                
                inputs = ai_models.qwen_tokenizer(prompt, return_tensors="pt").to(ai_models.qwen_model.device)
                
                import torch
                with torch.no_grad():
                    outputs = ai_models.qwen_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                        eos_token_id=ai_models.qwen_tokenizer.eos_token_id,
                        pad_token_id=ai_models.qwen_tokenizer.pad_token_id
                    )
                
                generated_text = ai_models.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # "assistant" 이후 텍스트만 가져오기
                if "assistant" in generated_text:
                    summary = generated_text.split("assistant")[-1].strip()
                else:
                    summary = generated_text[len(prompt):].strip()
                
                return summary if summary else text[:150] + "..."
                
            except Exception as e:
                print(f"[⚠️ Qwen 이메일 요약 실패] {str(e)}")
        
        # Qwen 실패 시 간단한 요약
        sentences = text.split('.')
        important_sentences = [s.strip() for s in sentences[:2] if len(s.strip()) > 10]
        return '. '.join(important_sentences) + '.' if important_sentences else text[:150] + "..."
        
    except Exception as e:
        print(f"[⚠️ 이메일 요약 오류] {str(e)}")
        return text[:150] + "..." if len(text) > 150 else text




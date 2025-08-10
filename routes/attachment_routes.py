from flask import Blueprint, request, jsonify

def create_attachment_routes(attachment_service, session_manager):
    attachment_bp = Blueprint('attachment', __name__)
    
    @attachment_bp.route('/api/attachment-info', methods=['POST'])
    def get_attachment_info():
        """특정 메일의 첨부파일 상세 정보 반환"""
        try:
            data = request.get_json()
            email_id = data.get("email_id")
            user_email = data.get("email", "")
            
            # ✅ 세션 확인 개선
            print(f"[📎 첨부파일 정보 요청] 사용자: {user_email}, 이메일ID: {email_id}")
            
            if not user_email:
                print("[❗인증 실패] 사용자 이메일이 없습니다.")
                return jsonify({"error": "사용자 이메일이 필요합니다."}), 400
            
            if not session_manager.session_exists(user_email):
                print(f"[❗인증 실패] 세션이 존재하지 않음: {user_email}")
                # 세션 복원 시도
                user_key = session_manager.get_user_key(user_email)
                saved_data = session_manager.load_user_session_from_file(user_email)
                
                if saved_data:
                    # 세션 복원
                    session_manager.user_sessions[user_key] = {
                        'email': user_email,
                        'extracted_todos': saved_data.get('extracted_todos', []),
                        'last_emails': saved_data.get('last_emails', []),
                        'login_time': saved_data.get('login_time')
                    }
                    print(f"[✅ 세션 복원] {user_email}")
                else:
                    return jsonify({"error": "로그인이 필요합니다. 다시 로그인해주세요."}), 401
            
            # 세션에서 해당 메일 찾기
            user_session = session_manager.get_session(user_email)
            last_emails = user_session.get('last_emails', [])
            target_email = None
            
            for email_data in last_emails:
                if str(email_data.get('id')) == str(email_id):
                    target_email = email_data
                    break
            
            if not target_email:
                print(f"[❗메일 찾기 실패] 이메일 ID {email_id}를 찾을 수 없음")
                return jsonify({"error": "해당 메일을 찾을 수 없습니다."}), 404
            
            attachments = target_email.get('attachments', [])
            print(f"[📎 첨부파일 조회 성공] {len(attachments)}개 첨부파일")
            
            return jsonify({
                "success": True,
                "email_id": email_id,
                "subject": target_email.get('subject', ''),
                "attachments": attachments,
                "attachment_count": len(attachments),
                "has_yolo_detections": any(att.get('yolo_detections') for att in attachments)
            })
            
        except Exception as e:
            print(f"[❗첨부파일 정보 오류] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @attachment_bp.route('/api/document-summary', methods=['POST'])
    def get_document_summary():
        """특정 첨부파일의 상세 문서 요약 반환 - PDF 요약 개선"""
        try:
            data = request.get_json()
            email_id = data.get("email_id")
            filename = data.get("filename", "")
            user_email = data.get("email", "")
            
            print(f"[📄 문서 요약 요청] 사용자: {user_email}, 이메일ID: {email_id}, 파일명: {filename}")
            
            if not user_email:
                print("[❗인증 실패] 사용자 이메일이 없습니다.")
                return jsonify({"error": "사용자 이메일이 필요합니다."}), 400
                
            if not email_id:
                print("[❗요청 실패] 이메일 ID가 없습니다.")
                return jsonify({"error": "이메일 ID가 필요합니다."}), 400
                
            if not filename:
                print("[❗요청 실패] 파일명이 없습니다.")
                return jsonify({"error": "파일명이 필요합니다."}), 400
            
            # 세션 존재 확인 및 복원 시도
            if not session_manager.session_exists(user_email):
                print(f"[❗세션 없음] {user_email} - 복원 시도")
                
                user_key = session_manager.get_user_key(user_email)
                saved_data = session_manager.load_user_session_from_file(user_email)
                
                if saved_data:
                    session_manager.user_sessions[user_key] = {
                        'email': user_email,
                        'extracted_todos': saved_data.get('extracted_todos', []),
                        'last_emails': saved_data.get('last_emails', []),
                        'login_time': saved_data.get('login_time')
                    }
                    print(f"[✅ 세션 복원 성공] {user_email}")
                else:
                    print(f"[❗세션 복원 실패] {user_email}")
                    return jsonify({
                        "error": "로그인이 필요합니다. 다시 로그인해주세요.",
                        "action": "reload_required"
                    }), 401
            
            # 세션에서 해당 메일의 첨부파일 찾기
            user_session = session_manager.get_session(user_email)
            last_emails = user_session.get('last_emails', [])
            target_attachment = None
            
            print(f"[🔍 메일 검색] 총 {len(last_emails)}개 메일에서 검색")
            
            for email_data in last_emails:
                if str(email_data.get('id')) == str(email_id):
                    print(f"[✅ 메일 발견] ID: {email_id}, 첨부파일: {len(email_data.get('attachments', []))}개")
                    for attachment in email_data.get('attachments', []):
                        if attachment.get('filename') == filename:
                            target_attachment = attachment
                            print(f"[✅ 첨부파일 발견] {filename}")
                            break
                    break
            
            if not target_attachment:
                print(f"[❗첨부파일 없음] 이메일 ID: {email_id}, 파일명: {filename}")
                return jsonify({"error": "해당 첨부파일을 찾을 수 없습니다."}), 404
            
            # 문서 요약 정보 반환 - PDF 처리 개선
            response_data = {
                "success": True,
                "filename": filename,
                "file_type": target_attachment.get('type', 'unknown'),
                "size": target_attachment.get('size', 0),
                "extraction_success": target_attachment.get('extraction_success', False)
            }
            
            # PDF 문서 처리 (document_pdf)
            if target_attachment.get('type') == 'document_pdf':
                extracted_text = target_attachment.get('extracted_text', '')
                
                # 요약 정보들을 다양한 키로 찾기
                summary_text = (
                    target_attachment.get('document_summary') or 
                    target_attachment.get('summary') or 
                    target_attachment.get('text_summary') or 
                    ''
                )
                
                # 요약이 없으면 즉석에서 생성
                if not summary_text and extracted_text:
                    print(f"[📝 즉석 요약 생성] {filename}")
                    summary_text = generate_quick_summary(extracted_text, filename)
                
                response_data.update({
                    "extracted_text": extracted_text[:2000] if extracted_text else "",  # 처음 2000자
                    "document_summary": summary_text,
                    "full_summary": summary_text,  # 전체 요약
                    "extraction_method": target_attachment.get('extraction_method', ''),
                    "text_length": target_attachment.get('text_length', 0),
                    "has_full_text": len(extracted_text) > 2000 if extracted_text else False,
                    "ocr_success": target_attachment.get('ocr_success', False),
                    "processing_method": target_attachment.get('processing_method', ''),
                    "preview_text": extracted_text[:500] if extracted_text else "",  # 미리보기용
                })
                
                print(f"[✅ PDF 요약 반환] {filename} - 텍스트: {len(extracted_text)}자, 요약: {len(summary_text)}자")
                
            # 이미지 파일 처리
            elif target_attachment.get('type') == 'image':
                yolo_detections = target_attachment.get('yolo_detections', [])
                detected_objects = target_attachment.get('detected_objects', [])
                
                # YOLO 결과 정리
                if yolo_detections and isinstance(yolo_detections[0], dict):
                    object_names = [det.get('class', 'unknown') for det in yolo_detections]
                elif yolo_detections:
                    object_names = yolo_detections
                elif detected_objects:
                    object_names = detected_objects
                else:
                    object_names = []
                
                response_data.update({
                    "yolo_detections": object_names,
                    "object_count": len(object_names),
                    "ocr_text": target_attachment.get('extracted_text', ''),
                    "text_summary": target_attachment.get('text_summary', ''),
                    "ocr_success": target_attachment.get('ocr_success', False),
                    "raw_yolo_data": yolo_detections
                })
                
                print(f"[🖼️ 이미지 정보] YOLO 객체: {len(object_names)}개, OCR: {target_attachment.get('ocr_success', False)}")
                
            # 기타 문서 타입 처리
            elif target_attachment.get('type', '').startswith('document_'):
                extracted_text = target_attachment.get('extracted_text', '')
                summary_text = (
                    target_attachment.get('document_summary') or 
                    target_attachment.get('summary') or 
                    target_attachment.get('text_summary') or 
                    ''
                )
                
                # 즉석 요약 생성
                if not summary_text and extracted_text:
                    summary_text = generate_quick_summary(extracted_text, filename)
                
                response_data.update({
                    "extracted_text": extracted_text[:2000] if extracted_text else "",
                    "document_summary": summary_text,
                    "extraction_method": target_attachment.get('extraction_method', ''),
                    "full_text_available": len(extracted_text) > 2000 if extracted_text else False,
                    "text_length": len(extracted_text) if extracted_text else 0
                })
                
                # 파일 타입별 추가 정보
                if target_attachment.get('slide_count'):
                    response_data['slide_count'] = target_attachment['slide_count']
                if target_attachment.get('sheet_info'):
                    response_data['sheet_info'] = target_attachment['sheet_info']
            
            print(f"[✅ 문서 요약 반환] {filename} - 타입: {response_data['file_type']}")
            return jsonify(response_data)
            
        except Exception as e:
            print(f"[❗문서 요약 API 오류] {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    def generate_quick_summary(text, filename):
        """빠른 문서 요약 생성 - DB 족보 특화"""
        try:
            if not text or len(text.strip()) < 50:
                return "텍스트가 너무 짧아 요약할 수 없습니다."
            
            lines = text.split('\n')
            important_lines = []
            
            # DB 관련 키워드
            db_keywords = [
                '문제', '답', '해답', '정답', '점수', '총점', 
                '기말고사', '중간고사', '시험', '과제',
                '데이터베이스', 'database', 'sql', 'query', 'select', 'join',
                '정규화', 'normalization', '1nf', '2nf', '3nf', 'bcnf',
                '관계', 'relation', 'table', 'entity', 'attribute',
                '트랜잭션', 'transaction', 'acid', 'commit', 'rollback',
                '인덱스', 'index', 'primary key', 'foreign key'
            ]
            
            # 중요한 라인 찾기
            for line in lines:
                line = line.strip()
                if len(line) > 10:
                    # DB 키워드 포함 라인
                    if any(keyword in line.lower() for keyword in db_keywords):
                        important_lines.append(line)
                    # 번호로 시작하는 문제 라인
                    elif line.match(r'^\d+\.') or line.startswith('문제') or line.startswith('Q'):
                        important_lines.append(line)
                    
                    if len(important_lines) >= 8:  # 최대 8줄
                        break
            
            if important_lines:
                summary = f"📋 {filename} 주요 내용:\n\n"
                for i, line in enumerate(important_lines, 1):
                    # 라인이 너무 길면 자르기
                    display_line = line[:150] + "..." if len(line) > 150 else line
                    summary += f"{i}. {display_line}\n"
                return summary
            else:
                # 키워드가 없으면 첫 부분 요약
                first_lines = [line.strip() for line in lines[:5] if line.strip() and len(line.strip()) > 10]
                if first_lines:
                    summary = f"📄 {filename} 시작 부분:\n\n"
                    for i, line in enumerate(first_lines, 1):
                        display_line = line[:150] + "..." if len(line) > 150 else line
                        summary += f"{i}. {display_line}\n"
                    return summary
                else:
                    return f"📄 {filename}: 문서를 분석했지만 의미있는 요약을 생성할 수 없습니다."
                    
        except Exception as e:
            print(f"[❗요약 생성 오류] {filename}: {str(e)}")
            return f"📄 {filename}: 요약 생성 중 오류가 발생했습니다."
    
    @attachment_bp.route('/api/clear-cache', methods=['POST'])
    def clear_attachment_cache():
        """첨부파일 캐시 초기화"""
        try:
            cache_count = attachment_service.clear_cache()
            
            return jsonify({
                "success": True,
                "message": f"캐시 {cache_count}개 항목이 삭제되었습니다.",
                "cleared_items": cache_count
            })
            
        except Exception as e:
            print(f"[❗캐시 초기화 오류] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return attachment_bp

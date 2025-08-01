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
            
            # 사용자 세션 확인
            if not session_manager.session_exists(user_email):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            # 세션에서 해당 메일 찾기
            user_session = session_manager.get_session(user_email)
            last_emails = user_session.get('last_emails', [])
            target_email = None
            
            for email_data in last_emails:
                if email_data.get('id') == email_id:
                    target_email = email_data
                    break
            
            if not target_email:
                return jsonify({"error": "해당 메일을 찾을 수 없습니다."}), 404
            
            attachments = target_email.get('attachments', [])
            
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
        """특정 첨부파일의 상세 문서 요약 반환"""
        try:
            data = request.get_json()
            email_id = data.get("email_id")
            filename = data.get("filename", "")
            user_email = data.get("email", "")
            
            # 사용자 세션 확인
            if not session_manager.session_exists(user_email):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            # 세션에서 해당 메일의 첨부파일 찾기
            user_session = session_manager.get_session(user_email)
            last_emails = user_session.get('last_emails', [])
            target_attachment = None
            
            for email_data in last_emails:
                if email_data.get('id') == email_id:
                    for attachment in email_data.get('attachments', []):
                        if attachment.get('filename') == filename:
                            target_attachment = attachment
                            break
                    break
            
            if not target_attachment:
                return jsonify({"error": "해당 첨부파일을 찾을 수 없습니다."}), 404
            
            # 문서 요약 정보 반환
            response_data = {
                "success": True,
                "filename": filename,
                "file_type": target_attachment.get('type', 'unknown'),
                "size": target_attachment.get('size', 0),
                "extraction_success": target_attachment.get('extraction_success', False)
            }
            
            # 타입별 상세 정보 추가
            if target_attachment.get('type') == 'image':
                response_data.update({
                    "yolo_detections": target_attachment.get('detected_objects', []),
                    "object_count": target_attachment.get('object_count', 0),
                    "ocr_text": target_attachment.get('extracted_text', ''),
                    "text_summary": target_attachment.get('text_summary', '')
                })
            
            elif target_attachment.get('type', '').startswith('document_'):
                response_data.update({
                    "extracted_text": target_attachment.get('extracted_text', '')[:1000],  # 처음 1000자만
                    "document_summary": target_attachment.get('document_summary', ''),
                    "extraction_method": target_attachment.get('extraction_method', ''),
                    "full_text_available": len(target_attachment.get('extracted_text', '')) > 1000
                })
                
                # 파일 타입별 추가 정보
                if target_attachment.get('pages'):
                    response_data['pages'] = target_attachment['pages']
                if target_attachment.get('slides'):
                    response_data['slides'] = target_attachment['slides']
                if target_attachment.get('sheets'):
                    response_data['sheets'] = target_attachment['sheets']
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"[❗문서 요약 API 오류] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
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
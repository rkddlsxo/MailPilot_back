from flask import Blueprint, request, jsonify

def create_chatbot_routes(chatbot_service, session_manager):
    chatbot_bp = Blueprint('chatbot', __name__)
    
    @chatbot_bp.route('/api/chatbot', methods=['POST'])
    def chatbot():
        """챗봇 대화 처리"""
        try:
            data = request.get_json()
            user_input = data.get("user_input", "").strip()
            user_email = data.get("email", "")
            app_password = data.get("app_password", "")
            
            if not user_input:
                return jsonify({"error": "입력이 비어있습니다."}), 400
            
            # 사용자 세션 확인
            if not session_manager.session_exists(user_email):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            # 챗봇 서비스로 처리 위임
            response, status_code = chatbot_service.process_user_input(user_input, user_email, app_password)
            
            return jsonify(response), status_code
            
        except Exception as e:
            print(f"[❗챗봇 라우트 오류] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @chatbot_bp.route('/api/generate-ai-reply', methods=['POST'])
    def generate_ai_reply():
        """AI 답장 생성"""
        try:
            data = request.get_json()
            sender = data.get('sender', '')
            subject = data.get('subject', '')
            body = data.get('body', '')
            current_user_email = data.get('email', '')
            
            if not all([sender, subject, body, current_user_email]):
                return jsonify({'error': '발신자, 제목, 본문, 사용자 이메일이 모두 필요합니다.'}), 400
            
            # 사용자 세션 확인
            if not session_manager.session_exists(current_user_email):
                return jsonify({'error': '로그인이 필요합니다.'}), 401
            
            # 챗봇 서비스로 처리 위임
            response, status_code = chatbot_service.generate_ai_reply(sender, subject, body, current_user_email)
            
            return jsonify(response), status_code
            
        except Exception as e:
            print(f"[❗AI 답장 라우트 오류] {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return chatbot_bp
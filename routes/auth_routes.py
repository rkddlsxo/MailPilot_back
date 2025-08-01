from flask import Blueprint, request, jsonify, session
import uuid

def create_auth_routes(session_manager):
    auth_bp = Blueprint('auth', __name__)
    
    @auth_bp.route('/api/login', methods=['POST'])
    def login_user():
        """사용자 로그인"""
        try:
            data = request.get_json()
            email = data.get('email', '')
            
            if not email:
                return jsonify({'error': '이메일이 필요합니다.'}), 400
            
            # 이전 세션 정리
            session_manager.clear_user_session(email)
            
            # 새 세션 ID 생성
            session_id = str(uuid.uuid4())
            
            # 세션 생성 또는 복원
            result = session_manager.create_or_restore_session(email, session_id)
            
            return jsonify({
                'success': True,
                'message': result['message'],
                'session_id': session_id,
                'restored_todos': result['todos_count']
            })
            
        except Exception as e:
            print(f"[❗로그인 실패] {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @auth_bp.route('/api/logout', methods=['POST'])
    def logout_user():
        """사용자 로그아웃"""
        try:
            data = request.get_json()
            email = data.get('email', '')
            
            if email:
                session_manager.clear_user_session(email)
                session.clear()
                
                return jsonify({
                    'success': True,
                    'message': '로그아웃 성공'
                })
            else:
                return jsonify({'error': '이메일이 필요합니다.'}), 400
                
        except Exception as e:
            print(f"[❗로그아웃 실패] {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return auth_bp
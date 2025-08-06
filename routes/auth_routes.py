from flask import Blueprint, request, jsonify, session
from models.tables import User
from models.db import db

from services.email_service import EmailService
from config import Config  # Gmail 서버 주소 등 포함된 설정
from transformers import pipeline  # 요약 모델

import uuid

def create_auth_routes(session_manager):
    auth_bp = Blueprint('auth', __name__)
    
    @auth_bp.route('/api/login', methods=['POST'])
    def login_user():
        """사용자 로그인"""
        try:
            data = request.get_json()
            email = data.get('email', '')
            app_password = data.get('app_password', '')
            
            if not email:
                return jsonify({'error': '이메일이 필요합니다.'}), 400
            
            # 이전 세션 정리
            session_manager.clear_user_session(email)
            
            # 새 세션 ID 생성
            session_id = str(uuid.uuid4())

            # ✅ DB에 사용자 등록 (없을 경우에만)
            if not User.query.filter_by(email=email).first():
                db.session.add(User(email=email))
                db.session.commit()
            
            # 세션 생성 또는 복원
            result = session_manager.create_or_restore_session(email, session_id)

            # ✅ 이메일 가져와서 DB에 저장 (요약 포함)
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            email_service = EmailService(config=Config(), summarizer=summarizer)
            
            print(f"[📬 로그인 중 메일 수집] {email}")
            fetched = email_service.fetch_emails(email, app_password, count=10)
            print(f"[📥 수신된 메일 수] {len(fetched)}")
            print(f"[🔐 받은 로그인 정보] 이메일: {email}, 앱 비번: {'***' if app_password else '(비어있음)'}")


            return jsonify({
                'success': True,
                'message': result['message'],
                'session_id': session_id,
                'restored_todos': result['todos_count'],
                'fetched_emails': len(fetched)

            })
            
        except Exception as e:
            print(f"[🔐 받은 로그인 정보] 이메일: {email}, 앱 비번: {'***' if app_password else '(비어있음)'}")

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
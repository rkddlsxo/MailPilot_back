from flask import Blueprint, request, jsonify, session
from models.tables import User
from models.db import db

from services.email_service import EmailService
from config import Config  # Gmail 서버 주소 등 포함된 설정
from transformers import pipeline  # 요약 모델
import torch

import uuid

def create_auth_routes(session_manager, ai_models=None):
    auth_bp = Blueprint('auth', __name__)
    
    @auth_bp.route('/api/login', methods=['POST'])
    def login_user():
        """사용자 로그인"""
        try:
            data = request.get_json()
            print(f"[🔍 받은 전체 데이터] {data}")
            
            email = data.get('email', '')
            app_password = data.get('app_password', '')
            
            print(f"[🔍 파싱된 데이터] 이메일: {email}, 앱 비번: {'***' if app_password else '(비어있음)'}")
            print(f"[🔍 앱 비번 길이] {len(app_password) if app_password else 0}")
            
            if not email:
                return jsonify({'error': '이메일이 필요합니다.'}), 400
            
            if not app_password:
                return jsonify({'error': '앱 비밀번호가 필요합니다.'}), 400
            
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

            # ✅ AI 모델만 초기화 (메일 처리는 별도 요청에서)
            print(f"[🔐 받은 로그인 정보] 이메일: {email}, 앱 비번: {'***' if app_password else '(비어있음)'}")
            print(f"[📬 로그인 완료] 메일 처리는 별도 요청에서 진행됩니다")


            return jsonify({
                'success': True,
                'message': result['message'],
                'session_id': session_id,
                'restored': result['restored']
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
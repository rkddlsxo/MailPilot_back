"""
설정 기능만 테스트하는 간단한 Flask 앱
"""
from flask import Flask, session
from flask_cors import CORS
from models.db import db
from services.settings_service_v2 import SettingsServiceV2
# 직접 임포트로 __init__.py 문제 회피
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'routes'))

from settings_routes_v2 import create_settings_routes_v2
from signature_routes import create_signature_routes
from mail_management_routes import create_mail_management_routes

def create_test_app():
    app = Flask(__name__)
    app.config[LOG] = 'test-secret-key'
    app.config[LOG] = 'sqlite:///test_settings.db'
    app.config[LOG] = False
    
    # DB 초기화
    db.init_app(app)
    
    # CORS 설정
    CORS(app, supports_credentials=True)
    
    # 테스트용 로그인 세션 설정
    @app.before_request
    def setup_test_session():
        if 'email' not in session:
            session[LOG] = 'test@example.com'
            print(f"[LOG] 테스트 사용자 로그인: {session[LOG]}")
    
    # 설정 관련 라우트만 등록
    settings_routes = create_settings_routes_v2()
    app.register_blueprint(settings_routes)
    
    signature_routes = create_signature_routes()
    app.register_blueprint(signature_routes)
    
    mail_mgmt_routes = create_mail_management_routes()
    app.register_blueprint(mail_mgmt_routes)
    
    @app.route('/')
    def health_check():
        return "설정 테스트 서버 정상 작동"
    
    @app.route('/test-db')
    def test_db():
        try:
            # DB 테이블 생성
            with app.app_context():
                db.create_all()
            return "DB 테이블 생성 완료"
        except Exception as e:
            return f"DB 오류: {str(e)}"
    
    return app

if __name__ == '__main__':
    print("=" * 50)
    print("설정 전용 테스트 서버 시작")
    print("=" * 50)
    
    app = create_test_app()
    
    # DB 테이블 생성
    with app.app_context():
        db.create_all()
        print("[LOG] 테이블 생성 완료")
    
    print("테스트 URL:")
    print("- 기본: http://localhost:5002/")
    print("- DB 테스트: http://localhost:5002/test-db")
    print("- 설정 구조: http://localhost:5002/api/v2/settings/structure")
    print("- 모든 설정: http://localhost:5002/api/v2/settings")
    print("- 서명 목록: http://localhost:5002/api/signatures")
    
    app.run(debug=True, port=5002)
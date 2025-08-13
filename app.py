from flask import Flask, jsonify
from flask_cors import CORS

from models.db import db  
from models.tables import User, Mail, Todo  # 앱 컨텍스트 안에서 사용 예정

# 모듈 임포트
from config import Config
from models.ai_models import AIModels
from models.user_session import UserSessionManager
from services.email_service import EmailService
from services.attachment_service import AttachmentService
from services.todo_service import TodoService
from services.chatbot_service import ChatbotService

# 라우트 임포트
from routes.auth_routes import create_auth_routes
from routes.email_routes import create_email_routes
from routes.todo_routes import create_todo_routes
from routes.chatbot_routes import create_chatbot_routes
from routes.attachment_routes import create_attachment_routes

def create_app():
    """Flask 애플리케이션 팩토리"""
    app = Flask(__name__)
    
    # 설정 로드
    config = Config()
    app.config.from_object(config)

     # SQLAlchemy 초기화
    db.init_app(app)
    
    # CORS 설정
    CORS(app, supports_credentials=True)
    
    # 모델 및 서비스 초기화
    print("[🔧 모델 및 서비스 초기화]")
    ai_models = AIModels(config)
    session_manager = UserSessionManager(config)
    email_service = EmailService(config, summarizer=ai_models.summarizer)
    attachment_service = AttachmentService(config, ai_models)
    todo_service = TodoService(config)
    chatbot_service = ChatbotService(config, ai_models, email_service)
    
    print("[🛣️ 라우트 등록]")
    # 라우트 등록
    auth_routes = create_auth_routes(session_manager, ai_models)
    app.register_blueprint(auth_routes)
    
    email_routes = create_email_routes(email_service, ai_models, session_manager, attachment_service, todo_service)
    app.register_blueprint(email_routes)
    
    todo_routes = create_todo_routes(session_manager, todo_service)
    app.register_blueprint(todo_routes)
    
    chatbot_routes = create_chatbot_routes(chatbot_service, session_manager)
    app.register_blueprint(chatbot_routes)
    
    attachment_routes = create_attachment_routes(attachment_service, session_manager)
    app.register_blueprint(attachment_routes)
    
    # 기본 라우트
    @app.route('/', methods=['GET'])
    def health_check():
        return "✅ 모듈화된 백엔드 정상 작동 중", 200
    
    @app.route('/api/session-info', methods=['GET'])
    def session_info():
        """현재 활성 세션 정보 반환"""
        return jsonify({
            "active_sessions": len(session_manager.user_sessions),
            "session_keys": [key[:8] + "..." for key in session_manager.user_sessions.keys()],
            "yolo_model_loaded": ai_models.yolo_model is not None,
            "qwen_model_loaded": ai_models.qwen_model is not None,
            "ocr_model_loaded": ai_models.ocr_reader is not None
        })
    
    @app.route('/api/test', methods=['POST'])
    def test():
        """테스트 엔드포인트"""
        from flask import request
        
        data = request.get_json()
        text = data.get("text", "")
        email = data.get("email", "")
        
        user_key = session_manager.get_user_key(email) if email else "anonymous"
        
        return jsonify({
            "message": f"✅ 모듈화된 백엔드 정상 작동: {text[:20]}...",
            "user_session": user_key[:8] + "..." if email else "no_session",
            "modules_loaded": {
                "ai_models": "✅",
                "session_manager": "✅", 
                "email_service": "✅",
                "attachment_service": "✅",
                "todo_service": "✅",
                "chatbot_service": "✅"
            }
        })
    
    return app

if __name__ == '__main__':
    

    print("=" * 60)
    print("🚀 모듈화된 메일 시스템 시작")
    print("=" * 60)
    
    app = create_app()
    
    with app.app_context():
        db.create_all()

    # YOLO 모델 미리 로딩 (선택적)
    print("[🔄 YOLO 모델 사전 로딩 시도...]")
    # ai_models.load_yolo_model()  # 필요시 주석 해제
    
    print("=" * 60)
    print("🌐 서버 시작: http://localhost:5001")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5001)

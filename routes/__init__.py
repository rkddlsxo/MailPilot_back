"""
Flask 라우트 모듈

이 패키지는 다음 라우트 팩토리들을 포함합니다:
- create_auth_routes: 인증 관련 라우트
- create_email_routes: 이메일 관련 라우트
- create_todo_routes: 할일 관리 라우트
- create_chatbot_routes: 챗봇 관련 라우트
- create_attachment_routes: 첨부파일 관련 라우트
"""

from .auth_routes import create_auth_routes
from .email_routes import create_email_routes
from .todo_routes import create_todo_routes
from .chatbot_routes import create_chatbot_routes
from .attachment_routes import create_attachment_routes

__all__ = [
    'create_auth_routes',
    'create_email_routes',
    'create_todo_routes', 
    'create_chatbot_routes',
    'create_attachment_routes'
]

__version__ = '1.0.0'
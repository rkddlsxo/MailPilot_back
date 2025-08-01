"""
비즈니스 로직 서비스 모듈

이 패키지는 다음 서비스들을 포함합니다:
- EmailService: IMAP/SMTP 이메일 처리
- AttachmentService: 첨부파일 + AI 분석 (YOLO, OCR)
- TodoService: 이메일에서 할일 추출
- ChatbotService: 챗봇 및 AI 기능 통합
"""

from .email_service import EmailService
from .attachment_service import AttachmentService
from .todo_service import TodoService
from .chatbot_service import ChatbotService

__all__ = [
    'EmailService',
    'AttachmentService', 
    'TodoService',
    'ChatbotService'
]

__version__ = '1.0.0'
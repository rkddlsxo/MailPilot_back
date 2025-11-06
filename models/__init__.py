"""
AI 모델 및 사용자 세션 관리 모듈

이 패키지는 다음을 포함합니다:
- AIModels: YOLO, Qwen, OCR 등 AI 모델 통합 관리
- UserSessionManager: 사용자 세션 및 파일 기반 데이터 저장
"""

from .ai_models import AIModels
from .user_session import UserSessionManager

# 패키지에서 직접 사용할 수 있는 클래스들
__all__ = [
    'AIModels',
    'UserSessionManager'
]

# 버전 정보
__version__ = '1.0.0'
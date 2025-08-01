# config.py - 설정 관리 (Nomic 토큰 오류 해결)

import os
from pathlib import Path

class Config:
    # Flask 설정
    SECRET_KEY = 'your-secret-key-here'
    
    # 디렉토리 설정
    BASE_DIR = Path(__file__).parent
    USER_DATA_DIR = BASE_DIR / "user_sessions"
    ATTACHMENT_FOLDER = "static/attachments"
    
    # AI 모델 설정
    YOLO_MODEL = 'yolov8n.pt'
    QWEN_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
    HUGGINGFACE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    
    # API 토큰 (환경변수에서 가져오기)
    NOMIC_TOKEN = os.getenv('NOMIC_TOKEN', 'your-nomic-token-here')  # 실제 토큰으로 교체
    HF_TOKEN = os.getenv('HF_TOKEN', 'your-hf-token-here')  # 실제 토큰으로 교체
    
    # 이메일 설정
    GMAIL_IMAP_SERVER = "imap.gmail.com"
    GMAIL_SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 465
    
    # 캐시 설정
    MAX_CACHE_SIZE = 100
    
    # 분류 라벨
    CANDIDATE_LABELS = [
        "university.",
        "spam mail.",
        "company.",
        "security alert."
    ]
    
    @classmethod
    def init_directories(cls):
        """필요한 디렉토리 생성"""
        cls.USER_DATA_DIR.mkdir(exist_ok=True)
        os.makedirs(cls.ATTACHMENT_FOLDER, exist_ok=True)
        
    @classmethod
    def validate_tokens(cls):
        """토큰 유효성 검사"""
        issues = []
        
        if cls.NOMIC_TOKEN == 'your-nomic-token-here':
            issues.append("NOMIC_TOKEN이 설정되지 않았습니다.")
            
        if cls.HF_TOKEN == 'your-hf-token-here':
            issues.append("HF_TOKEN이 설정되지 않았습니다.")
            
        return issues
# models/tables.py
from models.db import db
from datetime import datetime
import json

class User(db.Model):
    __tablename__ = 'user'
    email = db.Column(db.String(100), primary_key=True)

class Mail(db.Model):
    __tablename__ = 'mails'
    user_email = db.Column(db.String(100), db.ForeignKey('user.email'), primary_key=True)
    mail_id = db.Column(db.String(255), primary_key=True)
    subject = db.Column(db.Text)
    from_ = db.Column("from", db.String(255))
    body = db.Column(db.Text)
    raw_message = db.Column(db.Text(4294967295))
    date = db.Column(db.DateTime)
    summary = db.Column(db.Text)
    tag = db.Column(db.String(50))
    classification = db.Column(db.Text)
    attachments_data = db.Column(db.Text)
    mail_type = db.Column(db.String(10), default='inbox')  # 'inbox' 또는 'sent'

class Todo(db.Model):
    __tablename__ = 'todo'
    todo_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_email = db.Column(db.String(100), db.ForeignKey('user.email'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    event = db.Column(db.Text)
    date = db.Column(db.Date)
    time = db.Column(db.String(10))
    priority = db.Column(db.String(20), default='medium')
    status = db.Column(db.String(20), default='pending')
    mail_id = db.Column(db.String(255))


class Chatbot(db.Model):
    """학습된 명령어 패턴 저장"""
    __tablename__ = 'chatbot'
    
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(100), db.ForeignKey('user.email'), nullable=False)
    command = db.Column(db.Text, nullable=False)          # 사용자 입력 원문
    intent = db.Column(db.String(50), nullable=False)     # 처리 방식
    keywords = db.Column(db.Text)                         # JSON: 6개 키워드 지원
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    use_count = db.Column(db.Integer, default=1)          # 사용 횟수
    
    def get_keywords_dict(self):
        """keywords JSON을 딕셔너리로 반환"""
        if self.keywords:
            try:
                return json.loads(self.keywords)
            except:
                return {}
        return {}
    
    def set_keywords_dict(self, keywords_dict):
        """딕셔너리를 keywords JSON으로 저장"""
        self.keywords = json.dumps(keywords_dict, ensure_ascii=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'command': self.command,
            'intent': self.intent,
            'keywords': self.get_keywords_dict(),
            'use_count': self.use_count,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<Chatbot "{self.command}" -> {self.intent}>'


class UserSettings(db.Model):
    """계층적 설정 관리 모델 (카테고리-서브카테고리 구조)"""
    __tablename__ = 'user_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(255), db.ForeignKey('user.email'), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # GENERAL, MY_EMAIL
    subcategory = db.Column(db.String(50), nullable=False)  # READ, WRITE, AUTO_CLASSIFICATION 등
    settings_data = db.Column(db.JSON, default={})
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 복합 유니크 키 (한 사용자의 카테고리-서브카테고리 조합은 유일)
    __table_args__ = (
        db.UniqueConstraint('user_email', 'category', 'subcategory', name='unique_user_category_subcategory'),
    )
    
    def __repr__(self):
        return f'<UserSettings {self.user_email} - {self.category}/{self.subcategory}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_email': self.user_email,
            'category': self.category,
            'subcategory': self.subcategory,
            'settings': self.settings_data,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_or_create(cls, user_email, category, subcategory):
        """특정 카테고리/서브카테고리 설정 가져오기 또는 생성"""
        settings = cls.query.filter_by(
            user_email=user_email,
            category=category,
            subcategory=subcategory
        ).first()
        
        if not settings:
            settings = cls(
                user_email=user_email,
                category=category,
                subcategory=subcategory,
                settings_data=cls.get_default_settings(category, subcategory)
            )
            db.session.add(settings)
            db.session.commit()
        return settings
    
    @classmethod
    def get_default_settings(cls, category, subcategory):
        """카테고리별 기본 설정값"""
        # 새로운 상세 설정 구조
        defaults = {
            'GENERAL': {
                'READ': {
                    # 메일 가져오기 설정
                    'gmailFetchCount': 5,
                    'itemsPerPage': 10
                },
                'WRITE': {
                    # 기본 폰트
                    'fontFamily': 'system',
                    'fontSize': '14px',
                    # 보내는 사람 정보
                    'senderName': ''
                },
                'THEME': {
                    # 테마 설정
                    'appearance': 'light'
                }
            },
            'MY_EMAIL': {
                'SIGNATURE_MANAGEMENT': {
                    'signatures': [
                        # 기본 서명 하나 추가
                        {
                            'id': 1,
                            'name': '기본 서명',
                            'content': '',
                            'html_content': '',
                            'is_html': False,
                            'created_at': None,
                            'updated_at': None
                        }
                    ],
                    'next_id': 2  # 다음 서명 ID
                }
            }
        }
        
        return defaults.get(category, {}).get(subcategory, {})
    
    @classmethod
    def get_user_all_settings(cls, user_email):
        """사용자의 모든 설정 가져오기"""
        settings = cls.query.filter_by(user_email=user_email).all()
        
        print(f"[💾 DB] {user_email}의 저장된 설정 레코드 수: {len(settings)}")
        for setting in settings:
            print(f"[💾 DB] {setting.category}/{setting.subcategory}: {setting.settings_data}")
        
        result = {
            'GENERAL': {},
            'MY_EMAIL': {}
        }
        
        for setting in settings:
            if setting.category in result:
                result[setting.category][setting.subcategory] = setting.settings_data
        
        # 없는 설정은 기본값으로 채우기 (기존 설정과 병합)
        for category in ['GENERAL', 'MY_EMAIL']:
            for subcategory in cls.get_subcategories(category):
                default_settings = cls.get_default_settings(category, subcategory)
                if subcategory not in result[category]:
                    # 서브카테고리가 없으면 기본값 사용
                    result[category][subcategory] = default_settings
                else:
                    # 서브카테고리가 있으면 기본값과 병합 (저장된 값이 우선)
                    stored_settings = result[category][subcategory]
                    merged_settings = default_settings.copy()
                    
                    # 저장된 값들로 기본값 덮어쓰기 (None이 아닌 값만)
                    for key, value in stored_settings.items():
                        if value is not None:
                            merged_settings[key] = value
                    
                    result[category][subcategory] = merged_settings
        
        return result
    
    @classmethod
    def get_subcategories(cls, category):
        """카테고리별 서브카테고리 목록"""
        subcategories = {
            'GENERAL': ['READ', 'WRITE', 'THEME'],
            'MY_EMAIL': ['SIGNATURE_MANAGEMENT']
        }
        return subcategories.get(category, [])
    
    def update_settings(self, data):
        """설정 업데이트"""
        if self.settings_data:
            self.settings_data.update(data)
        else:
            self.settings_data = data
        self.updated_at = datetime.utcnow()
        db.session.commit()
        return True

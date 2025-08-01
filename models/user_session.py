import json
import hashlib
from datetime import datetime
from pathlib import Path

class UserSessionManager:
    def __init__(self, config):
        self.config = config
        self.user_sessions = {}
        self.config.init_directories()
    
    def get_user_key(self, email):
        """이메일 기반 사용자 키 생성"""
        return hashlib.md5(email.encode()).hexdigest()
    
    def get_user_file_path(self, user_email):
        """사용자별 데이터 파일 경로"""
        user_hash = hashlib.md5(user_email.encode()).hexdigest()[:16]
        return self.config.USER_DATA_DIR / f"user_{user_hash}.json"
    
    def save_user_session_to_file(self, user_email):
        """현재 세션을 파일에 저장"""
        try:
            user_key = self.get_user_key(user_email)
            if user_key not in self.user_sessions:
                return False
                
            file_path = self.get_user_file_path(user_email)
            session_data = self.user_sessions[user_key]
            
            save_data = {
                'user_email': user_email,
                'extracted_todos': session_data.get('extracted_todos', []),
                'last_emails': session_data.get('last_emails', []),
                'last_update': datetime.now().isoformat(),
                'login_time': session_data.get('login_time'),
                'session_id': session_data.get('session_id')
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            todos_count = len(session_data.get('extracted_todos', []))
            print(f"[💾 세션 저장] {user_email}: {todos_count}개 할일")
            return True
            
        except Exception as e:
            print(f"[❗저장 실패] {user_email}: {str(e)}")
            return False
    
    def load_user_session_from_file(self, user_email):
        """파일에서 세션 데이터 로드"""
        try:
            file_path = self.get_user_file_path(user_email)
            
            if not file_path.exists():
                print(f"[📁 새 사용자] {user_email}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('user_email') == user_email:
                todos_count = len(data.get('extracted_todos', []))
                print(f"[📂 세션 복원] {user_email}: {todos_count}개 할일")
                return data
                
            return None
            
        except Exception as e:
            print(f"[❗로드 실패] {user_email}: {str(e)}")
            return None
    
    def clear_user_session(self, email):
        """특정 사용자의 세션 정리"""
        user_key = self.get_user_key(email)
        if user_key in self.user_sessions:
            self.save_user_session_to_file(email)
            del self.user_sessions[user_key]
            print(f"[🗑️ 세션 정리] {email} - 파일 저장 후 메모리 정리")
    
    def create_or_restore_session(self, email, session_id):
        """세션 생성 또는 복원"""
        user_key = self.get_user_key(email)
        saved_data = self.load_user_session_from_file(email)
        
        if saved_data:
            # 파일에서 복원
            self.user_sessions[user_key] = {
                'email': email,
                'session_id': session_id,
                'extracted_todos': saved_data.get('extracted_todos', []),
                'last_emails': saved_data.get('last_emails', []),
                'login_time': datetime.now().isoformat()
            }
            
            todos_count = len(saved_data.get('extracted_todos', []))
            return {
                'restored': True,
                'todos_count': todos_count,
                'message': f'로그인 성공 - {todos_count}개 할일 복원됨'
            }
        else:
            # 새 세션 생성
            self.user_sessions[user_key] = {
                'email': email,
                'session_id': session_id,
                'last_emails': [],
                'extracted_todos': [],
                'login_time': datetime.now().isoformat()
            }
            
            return {
                'restored': False,
                'todos_count': 0,
                'message': '로그인 성공 - 새 세션'
            }
    
    def get_session(self, email):
        """사용자 세션 가져오기"""
        user_key = self.get_user_key(email)
        return self.user_sessions.get(user_key)
    
    def session_exists(self, email):
        """세션 존재 여부 확인"""
        user_key = self.get_user_key(email)
        return user_key in self.user_sessions
"""
계층적 설정 관리 서비스
"""
from models.tables import UserSettings
from models.db import db
from flask import jsonify
from datetime import datetime

class SettingsService:
    """계층적 설정 관리 서비스"""
    
    def __init__(self):
        self.categories = {
            'GENERAL': ['READ', 'WRITE', 'THEME'],
            'MY_EMAIL': ['SIGNATURE_MANAGEMENT']
        }
    
    def get_all_settings(self, user_email):
        """사용자의 모든 설정 가져오기"""
        try:
            print(f"[⚙️ 설정] {user_email} 사용자의 모든 설정 조회 시작")
            settings = UserSettings.get_user_all_settings(user_email)
            print(f"[⚙️ 설정] 설정 카테고리: {list(settings.keys())}")
            for category, subcategories in settings.items():
                print(f"[⚙️ 설정] {category}: {list(subcategories.keys())}")
            return {
                'success': True,
                'settings': settings
            }
        except Exception as e:
            print(f"[❌ 설정] 전체 설정 불러오기 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_category_settings(self, user_email, category):
        """특정 카테고리의 모든 설정 가져오기"""
        try:
            print(f"[⚙️ 설정] {user_email} 사용자의 {category} 카테고리 설정 조회")
            if category not in self.categories:
                return {
                    'success': False,
                    'error': f'Invalid category: {category}'
                }
            
            result = {}
            for subcategory in self.categories[category]:
                setting = UserSettings.get_or_create(user_email, category, subcategory)
                result[subcategory] = setting.settings_data
            
            print(f"[⚙️ 설정] {category} 카테고리 설정 조회 완료")
            return {
                'success': True,
                'category': category,
                'settings': result
            }
        except Exception as e:
            print(f"[❌ 설정] 카테고리 설정 불러오기 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_specific_settings(self, user_email, category, subcategory):
        """특정 카테고리/서브카테고리 설정 가져오기"""
        try:
            print(f"[⚙️ 설정] {user_email} 사용자의 {category}/{subcategory} 설정 조회")
            setting = UserSettings.get_or_create(user_email, category, subcategory)
            print(f"[⚙️ 설정] {category}/{subcategory} 설정 조회 완료")
            return {
                'success': True,
                'category': category,
                'subcategory': subcategory,
                'settings': setting.settings_data
            }
        except Exception as e:
            print(f"[❌ 설정] 개별 설정 불러오기 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_settings(self, user_email, category, subcategory, data):
        """설정 업데이트"""
        try:
            print(f"[⚙️ 설정] {user_email}의 {category}/{subcategory} 설정 업데이트 요청")
            print(f"[⚙️ 설정] 업데이트 데이터: {data}")
            setting = UserSettings.get_or_create(user_email, category, subcategory)
            setting.update_settings(data)
            
            print(f"[✅ 설정] {user_email}의 {category}/{subcategory} 설정 업데이트 완료")
            return {
                'success': True,
                'category': category,
                'subcategory': subcategory,
                'settings': setting.settings_data
            }
        except Exception as e:
            print(f"[❌ 설정] 설정 업데이트 실패: {e}")
            db.session.rollback()
            return {
                'success': False,
                'error': str(e)
            }
    
    def reset_settings(self, user_email, category=None, subcategory=None):
        """설정 초기화"""
        try:
            print(f"[🔄 설정] {user_email} 사용자의 설정 초기화 시작")
            if category and subcategory:
                # 특정 서브카테고리만 초기화
                print(f"[🔄 설정] {category}/{subcategory} 초기화")
                setting = UserSettings.query.filter_by(
                    user_email=user_email,
                    category=category,
                    subcategory=subcategory
                ).first()
                
                if setting:
                    setting.settings_data = UserSettings.get_default_settings(category, subcategory)
                    setting.updated_at = datetime.utcnow()
                    db.session.commit()
                    
            elif category:
                # 카테고리 전체 초기화
                print(f"[🔄 설정] {category} 카테고리 전체 초기화")
                for subcat in self.categories.get(category, []):
                    setting = UserSettings.query.filter_by(
                        user_email=user_email,
                        category=category,
                        subcategory=subcat
                    ).first()
                    
                    if setting:
                        setting.settings_data = UserSettings.get_default_settings(category, subcat)
                        setting.updated_at = datetime.utcnow()
                
                db.session.commit()
                
            else:
                # 모든 설정 초기화
                print(f"[🔄 설정] 모든 설정 초기화")
                UserSettings.query.filter_by(user_email=user_email).delete()
                db.session.commit()
            
            print(f"[✅ 설정] 설정 초기화 완료")
            return {
                'success': True,
                'message': 'Settings reset successfully'
            }
            
        except Exception as e:
            print(f"[❌ 설정] 설정 초기화 실패: {e}")
            db.session.rollback()
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_setting_value(self, user_email, category, subcategory, key):
        """특정 설정값 하나만 가져오기"""
        try:
            print(f"[⚙️ 설정] {user_email}의 {category}/{subcategory}/{key} 값 조회")
            setting = UserSettings.get_or_create(user_email, category, subcategory)
            value = setting.settings_data.get(key)
            print(f"[⚙️ 설정] {key} 값: {value}")
            return {
                'success': True,
                'value': value
            }
        except Exception as e:
            print(f"[❌ 설정] 설정값 조회 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def set_setting_value(self, user_email, category, subcategory, key, value):
        """특정 설정값 하나만 설정"""
        try:
            print(f"[⚙️ 설정] {user_email}의 {category}/{subcategory}/{key} 값 설정: {value}")
            setting = UserSettings.get_or_create(user_email, category, subcategory)
            
            print(f"[📝 설정] 기존 설정 데이터: {setting.settings_data}")
            
            if not setting.settings_data:
                setting.settings_data = {}
            
            # JSON 필드를 직접 수정하면 SQLAlchemy가 변경을 감지하지 못할 수 있으므로
            # 새로운 딕셔너리를 만들어서 할당
            new_settings_data = setting.settings_data.copy()
            new_settings_data[key] = value
            setting.settings_data = new_settings_data
            setting.updated_at = datetime.utcnow()
            
            print(f"[📝 설정] 업데이트될 설정 데이터: {setting.settings_data}")
            
            db.session.commit()
            
            print(f"[✅ 설정] {key} 값 설정 완료")
            return {
                'success': True,
                'message': f'{key} updated successfully'
            }
        except Exception as e:
            print(f"[❌ 설정] 설정값 설정 실패: {e}")
            db.session.rollback()
            return {
                'success': False,
                'error': str(e)
            }
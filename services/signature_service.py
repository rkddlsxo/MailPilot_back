"""
서명 관리 서비스
"""
from datetime import datetime
from models.tables import UserSettings
from models.db import db

class SignatureService:
    """서명 관리 서비스"""
    
    @staticmethod
    def get_signatures(user_email):
        """사용자의 모든 서명 가져오기"""
        try:
            print(f"[📝 서명] {user_email} 사용자의 서명 목록 조회")
            settings = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'SIGNATURE_MANAGEMENT')
            signatures = settings.settings_data.get('signatures', [])
            print(f"[📝 서명] 총 {len(signatures)}개의 서명 발견")
            return {
                'success': True,
                'signatures': signatures,
                'next_id': settings.settings_data.get('next_id', 1)
            }
        except Exception as e:
            print(f"[❌ 서명] 서명 조회 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def add_signature(user_email, name, content, html_content='', is_html=False):
        """새 서명 추가"""
        try:
            print(f"[📝 서명] {user_email} 사용자 새 서명 추가 요청: '{name}'")
            settings = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'SIGNATURE_MANAGEMENT')
            
            signatures = settings.settings_data.get('signatures', [])
            next_id = settings.settings_data.get('next_id', 1)
            print(f"[📝 서명] 현재 서명 수: {len(signatures)}, 새 ID: {next_id}")
            
            new_signature = {
                'id': next_id,
                'name': name,
                'content': content,
                'html_content': html_content,
                'is_html': is_html,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            signatures.append(new_signature)
            
            settings.settings_data['signatures'] = signatures
            settings.settings_data['next_id'] = next_id + 1
            settings.updated_at = datetime.utcnow()
            
            db.session.commit()
            print(f"[✅ 서명] 서명 추가 완료: ID {next_id}, 이름 '{name}'")
            
            return {
                'success': True,
                'signature': new_signature,
                'message': '서명이 추가되었습니다.'
            }
        except Exception as e:
            print(f"[❌ 서명] 서명 추가 실패: {e}")
            db.session.rollback()
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def update_signature(user_email, signature_id, name=None, content=None, html_content=None, is_html=None):
        """서명 수정"""
        try:
            settings = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'SIGNATURE_MANAGEMENT')
            signatures = settings.settings_data.get('signatures', [])
            
            for signature in signatures:
                if signature['id'] == signature_id:
                    if name is not None:
                        signature['name'] = name
                    if content is not None:
                        signature['content'] = content
                    if html_content is not None:
                        signature['html_content'] = html_content
                    if is_html is not None:
                        signature['is_html'] = is_html
                    signature['updated_at'] = datetime.utcnow().isoformat()
                    break
            else:
                return {'success': False, 'error': '서명을 찾을 수 없습니다.'}
            
            settings.settings_data['signatures'] = signatures
            settings.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            return {
                'success': True,
                'message': '서명이 수정되었습니다.'
            }
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def delete_signature(user_email, signature_id):
        """서명 삭제"""
        try:
            settings = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'SIGNATURE_MANAGEMENT')
            signatures = settings.settings_data.get('signatures', [])
            
            # 최소 하나의 서명은 유지
            if len(signatures) <= 1:
                return {'success': False, 'error': '최소 하나의 서명은 유지해야 합니다.'}
            
            original_length = len(signatures)
            signatures = [sig for sig in signatures if sig['id'] != signature_id]
            
            if len(signatures) == original_length:
                return {'success': False, 'error': '서명을 찾을 수 없습니다.'}
            
            settings.settings_data['signatures'] = signatures
            settings.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            return {
                'success': True,
                'message': '서명이 삭제되었습니다.'
            }
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_signature_status(user_email):
        """서명 사용 상태 가져오기"""
        try:
            settings = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'SIGNATURE_MANAGEMENT')
            return {
                'success': True,
                'enabled': settings.settings_data.get('enabled', True),
                'default_signature': settings.settings_data.get('defaultSignature', 0)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def set_signature_status(user_email, enabled, default_signature=None):
        """서명 사용 상태 설정"""
        try:
            settings = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'SIGNATURE_MANAGEMENT')
            
            settings.settings_data['enabled'] = enabled
            if default_signature is not None:
                settings.settings_data['defaultSignature'] = default_signature
            settings.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            return {
                'success': True,
                'message': '서명 설정이 업데이트되었습니다.'
            }
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_active_signature(user_email):
        """현재 활성화된 서명 가져오기"""
        try:
            print(f"[📝 서명] {user_email} 활성 서명 조회 시작")
            
            # 서명 설정 가져오기
            settings = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'SIGNATURE_MANAGEMENT')
            
            # 서명 사용 여부 확인
            enabled = settings.settings_data.get('enabled', True)
            print(f"[📝 서명] 서명 사용 여부: {enabled}")
            
            if not enabled:
                print("[📝 서명] 서명이 비활성화됨")
                return {'success': True, 'signature': None}
            
            # 서명 목록 가져오기
            signatures = settings.settings_data.get('signatures', [])
            print(f"[📝 서명] 저장된 서명 수: {len(signatures)}")
            
            if signatures:
                # 첫 번째 서명 반환 (1개만 사용)
                active_signature = signatures[0]
                print(f"[📝 서명] 활성 서명 발견: {active_signature.get('name', 'Unknown')}")
                return {
                    'success': True,
                    'signature': active_signature
                }
            
            print("[📝 서명] 저장된 서명 없음")
            return {'success': True, 'signature': None}
        except Exception as e:
            print(f"[❌ 서명] 활성 서명 조회 실패: {e}")
            return {'success': False, 'error': str(e)}
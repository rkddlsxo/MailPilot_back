"""
설정을 실제 이메일 서비스에 적용하는 서비스
"""
from models.tables import UserSettingsV2
from models.db import db

class SettingsApplyService:
    """설정 적용 서비스"""
    
    @staticmethod
    def get_read_settings(user_email):
        """읽기 설정 가져오기"""
        settings = UserSettingsV2.get_or_create(user_email, 'GENERAL', 'READ')
        return settings.settings_data
    
    @staticmethod
    def get_write_settings(user_email):
        """쓰기 설정 가져오기"""
        settings = UserSettingsV2.get_or_create(user_email, 'GENERAL', 'WRITE')
        return settings.settings_data
    
    @staticmethod
    def apply_list_view_settings(user_email):
        """목록 뷰 설정 적용"""
        settings = SettingsApplyService.get_read_settings(user_email)
        
        return {
            'itemsPerPage': settings.get('itemsPerPage', 50),
            'popupInListView': settings.get('popupInListView', False),
            'autoSelectFirstMail': settings.get('autoSelectFirstMail', True)
        }
    
    @staticmethod
    def apply_mail_view_settings(user_email):
        """메일 보기 설정 적용"""
        settings = SettingsApplyService.get_read_settings(user_email)
        
        return {
            'externalContent': 'always',  # 항상 외부 콘텐츠 표시
            'afterAction': settings.get('afterAction', 'toList')
        }
    
    @staticmethod
    def apply_compose_settings(user_email):
        """메일 작성 설정 적용"""
        settings = SettingsApplyService.get_write_settings(user_email)
        
        compose_settings = {
            'editorType': settings.get('editorType', 'html'),
            'fontFamily': settings.get('fontFamily', '기본글꼴'),
            'fontSize': settings.get('fontSize', '14px'),
            'senderEmail': settings.get('senderEmail', user_email),
            'senderName': settings.get('senderName', ''),
            'showInCompose': settings.get('showInCompose', True),
            'includeMe': settings.get('includeMe', 'none'),
            'individualSend': settings.get('individualSend', 'disabled'),
            'previewMode': settings.get('previewMode', 'none'),
            'previewConditions': settings.get('previewConditions', []),
            'delayedSend': settings.get('delayedSend', 'disabled'),
            'delayMinutes': settings.get('delayMinutes', 5)
        }
        
        # 보내는 이메일이 비어있으면 사용자 이메일 사용
        if not compose_settings['senderEmail']:
            compose_settings['senderEmail'] = user_email
            
        return compose_settings
    
    @staticmethod
    def should_preview_mail(user_email, mail_data):
        """메일 미리보기 여부 결정"""
        settings = SettingsApplyService.get_write_settings(user_email)
        preview_mode = settings.get('previewMode', 'none')
        
        if preview_mode == 'none':
            return False
        elif preview_mode == 'all':
            return True
        elif preview_mode == 'important':
            # 중요 메일 체크 로직
            conditions = settings.get('previewConditions', [])
            
            # 중요 메일(!) 체크
            if 'importantMail' in conditions and mail_data.get('important'):
                return True
            
            # 외부 수신인 체크
            if 'externalRecipient' in conditions:
                recipients = mail_data.get('recipients', [])
                sender_domain = user_email.split('@')[1]
                for recipient in recipients:
                    if '@' in recipient and recipient.split('@')[1] != sender_domain:
                        return True
        
        return False
    
    @staticmethod
    def get_delayed_send_time(user_email):
        """대기 발송 시간 가져오기 (분 단위)"""
        settings = SettingsApplyService.get_write_settings(user_email)
        
        if settings.get('delayedSend', 'disabled') == 'enabled':
            return settings.get('delayMinutes', 5)
        return 0
    
    @staticmethod
    def should_include_me_in_mail(user_email):
        """나를 포함할지 여부와 방법 반환"""
        settings = SettingsApplyService.get_write_settings(user_email)
        include_me = settings.get('includeMe', 'none')
        
        return {
            'include': include_me != 'none',
            'method': include_me  # 'cc', 'bcc', or 'none'
        }
    
    @staticmethod
    def get_signature_settings(user_email):
        """서명 설정 가져오기"""
        settings = UserSettingsV2.get_or_create(user_email, 'MY_EMAIL', 'SIGNATURE')
        return settings.settings_data
    
    @staticmethod
    def get_spam_settings(user_email):
        """스팸 설정 가져오기"""
        settings = UserSettingsV2.get_or_create(user_email, 'MY_EMAIL', 'SPAM_SETTINGS')
        return settings.settings_data
    
    @staticmethod
    def get_auto_classification_settings(user_email):
        """자동 분류 설정 가져오기"""
        settings = UserSettingsV2.get_or_create(user_email, 'MY_EMAIL', 'AUTO_CLASSIFICATION')
        return settings.settings_data
    
    @staticmethod
    def after_delete_action(user_email):
        """삭제 후 동작 가져오기"""
        settings = SettingsApplyService.get_read_settings(user_email)
        return settings.get('afterAction', 'toList')
    
    @staticmethod
    def should_show_external_content(user_email):
        """외부 콘텐츠 표시 여부"""
        # 항상 외부 콘텐츠 표시, 경고 없음
        return {
            'show': True,
            'confirm': False
        }
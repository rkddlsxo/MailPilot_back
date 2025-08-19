"""
메일 자동 삭제 서비스
"""
from datetime import datetime, timedelta
from models.tables import UserSettings, Mail
from models.db import db
from sqlalchemy import and_

class MailCleanupService:
    """메일 자동 삭제 서비스"""
    
    @staticmethod
    def get_deletion_settings(user_email):
        """메일 삭제 설정 가져오기"""
        try:
            print(f"[🗑️ 삭제] {user_email} 사용자의 메일 삭제 설정 조회")
            settings = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'MAIL_DELETE')
            print(f"[🗑️ 삭제] 설정 조회 완료: {settings.settings_data}")
            return {
                'success': True,
                'settings': settings.settings_data
            }
        except Exception as e:
            print(f"[❌ 삭제] 설정 조회 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def update_deletion_settings(user_email, settings_data):
        """메일 삭제 설정 업데이트"""
        try:
            print(f"[🗑️ 삭제] {user_email} 사용자의 메일 삭제 설정 업데이트")
            settings = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'MAIL_DELETE')
            settings.settings_data.update(settings_data)
            settings.updated_at = datetime.utcnow()
            
            db.session.commit()
            print(f"[✅ 삭제] 메일 삭제 설정 업데이트 완료")
            
            return {
                'success': True,
                'message': '메일 삭제 설정이 업데이트되었습니다.'
            }
        except Exception as e:
            print(f"[❌ 삭제] 설정 업데이트 실패: {e}")
            db.session.rollback()
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_period_days(period_setting):
        """기간 설정을 일 수로 변환"""
        period_map = {
            '1주일': 7,
            '2주일': 14,
            '1개월': 30,
            '3개월': 90,
            '6개월': 180,
            '1년': 365
        }
        return period_map.get(period_setting, 7)
    
    @staticmethod
    def cleanup_old_mails(user_email):
        """오래된 메일 자동 삭제 실행"""
        try:
            print(f"[🗑️ 삭제] {user_email} 사용자의 오래된 메일 자동 삭제 시작")
            settings_result = MailCleanupService.get_deletion_settings(user_email)
            if not settings_result['success']:
                return settings_result
            
            settings = settings_result['settings']
            period_days = MailCleanupService.get_period_days(settings.get('periodSetting', '1주일'))
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)
            
            print(f"[🗑️ 삭제] 삭제 기준일: {cutoff_date} ({period_days}일 전)")
            
            deleted_counts = {
                'sent': 0,
                'spam': 0,
                'trash': 0
            }
            
            # 보낸 메일함 자동 삭제
            if settings.get('autoDeleteSentMail', False):
                sent_mails = Mail.query.filter(
                    and_(
                        Mail.user_email == user_email,
                        Mail.mail_type == 'sent',
                        Mail.date < cutoff_date
                    )
                ).all()
                
                for mail in sent_mails:
                    db.session.delete(mail)
                deleted_counts['sent'] = len(sent_mails)
                print(f"[🗑️ 삭제] 보낸 메일 {len(sent_mails)}개 삭제 예정")
            
            # 스팸 메일함 자동 삭제
            if settings.get('autoDeleteSpamMail', False):
                spam_mails = Mail.query.filter(
                    and_(
                        Mail.user_email == user_email,
                        Mail.classification == 'spam mail.',
                        Mail.date < cutoff_date
                    )
                ).all()
                
                for mail in spam_mails:
                    db.session.delete(mail)
                deleted_counts['spam'] = len(spam_mails)
                print(f"[🗑️ 삭제] 스팸 메일 {len(spam_mails)}개 삭제 예정")
            
            # 휴지통 메일 자동 삭제
            if settings.get('autoDeleteTrashMail', False):
                trash_mails = Mail.query.filter(
                    and_(
                        Mail.user_email == user_email,
                        Mail.tag == '휴지통',
                        Mail.date < cutoff_date
                    )
                ).all()
                
                for mail in trash_mails:
                    db.session.delete(mail)
                deleted_counts['trash'] = len(trash_mails)
                print(f"[🗑️ 삭제] 휴지통 메일 {len(trash_mails)}개 삭제 예정")
            
            db.session.commit()
            print(f"[✅ 삭제] 자동 삭제 완료")
            
            return {
                'success': True,
                'deleted_counts': deleted_counts,
                'message': f"자동 삭제 완료: 보낸메일 {deleted_counts['sent']}개, 스팸 {deleted_counts['spam']}개, 휴지통 {deleted_counts['trash']}개"
            }
            
        except Exception as e:
            print(f"[❌ 삭제] 자동 삭제 실패: {e}")
            db.session.rollback()
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def preview_cleanup(user_email):
        """삭제 예상 메일 수 미리보기"""
        try:
            print(f"[👀 삭제] {user_email} 사용자의 삭제 예상 메일 수 미리보기")
            settings_result = MailCleanupService.get_deletion_settings(user_email)
            if not settings_result['success']:
                return settings_result
            
            settings = settings_result['settings']
            period_days = MailCleanupService.get_period_days(settings.get('periodSetting', '1주일'))
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)
            
            preview_counts = {
                'sent': 0,
                'spam': 0,
                'trash': 0
            }
            
            # 보낸 메일함 예상 삭제 수
            if settings.get('autoDeleteSentMail', False):
                preview_counts['sent'] = Mail.query.filter(
                    and_(
                        Mail.user_email == user_email,
                        Mail.mail_type == 'sent',
                        Mail.date < cutoff_date
                    )
                ).count()
            
            # 스팸 메일함 예상 삭제 수
            if settings.get('autoDeleteSpamMail', False):
                preview_counts['spam'] = Mail.query.filter(
                    and_(
                        Mail.user_email == user_email,
                        Mail.classification == 'spam mail.',
                        Mail.date < cutoff_date
                    )
                ).count()
            
            # 휴지통 예상 삭제 수
            if settings.get('autoDeleteTrashMail', False):
                preview_counts['trash'] = Mail.query.filter(
                    and_(
                        Mail.user_email == user_email,
                        Mail.tag == '휴지통',
                        Mail.date < cutoff_date
                    )
                ).count()
            
            print(f"[👀 삭제] 미리보기 완료: {preview_counts}")
            
            return {
                'success': True,
                'preview_counts': preview_counts,
                'cutoff_date': cutoff_date.isoformat(),
                'period_days': period_days
            }
            
        except Exception as e:
            print(f"[❌ 삭제] 미리보기 실패: {e}")
            return {'success': False, 'error': str(e)}
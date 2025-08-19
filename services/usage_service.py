"""
사용량 모니터링 서비스
"""
from datetime import datetime, timedelta
from sqlalchemy import and_, func
from models.tables import Mail, UserSettings
from models.db import db

class UsageService:
    """사용량 모니터링 서비스"""
    
    @staticmethod
    def calculate_mail_storage_usage(user_email):
        """메일 저장소 사용량 계산"""
        try:
            print(f"[📊 사용량] {user_email} 사용자의 메일 저장소 사용량 계산")
            
            # 사용자의 모든 메일 조회
            mails = Mail.query.filter_by(user_email=user_email).all()
            
            total_size_bytes = 0
            total_count = 0
            
            for mail in mails:
                # 메일 본문 크기 계산
                if mail.body:
                    total_size_bytes += len(mail.body.encode('utf-8'))
                if mail.subject:
                    total_size_bytes += len(mail.subject.encode('utf-8'))
                if mail.raw_message:
                    total_size_bytes += len(mail.raw_message.encode('utf-8'))
                
                total_count += 1
            
            # MB로 변환
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            print(f"[📊 사용량] 총 {total_count}개 메일, {total_size_mb:.2f}MB 사용")
            
            return {
                'success': True,
                'total_count': total_count,
                'total_size_bytes': total_size_bytes,
                'total_size_mb': round(total_size_mb, 2)
            }
        except Exception as e:
            print(f"[❌ 사용량] 저장소 사용량 계산 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_usage_statistics(user_email):
        """종합 사용량 통계 가져오기"""
        try:
            print(f"[📊 사용량] {user_email} 사용자의 사용량 통계 계산 시작")
            
            # 메일 저장소 사용량
            storage_result = UsageService.calculate_mail_storage_usage(user_email)
            if not storage_result['success']:
                return storage_result
            
            # 최근 30일 메일 통계
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_mails = Mail.query.filter(
                and_(
                    Mail.user_email == user_email,
                    Mail.date >= thirty_days_ago
                )
            ).all()
            
            # 분류별 통계
            classification_data = {}
            for mail in recent_mails:
                classification = mail.classification or 'unknown'
                classification_data[classification] = classification_data.get(classification, 0) + 1
            
            # 사용량 설정 가져오기
            settings_result = UserSettings.get_or_create(user_email, 'MY_EMAIL', 'USAGE')
            quota_warning = settings_result.settings_data.get('quotaWarning', 80) if settings_result else 80
            
            # 할당된 용량 (예: 1GB)
            allocated_quota_mb = 1024  # 1GB
            usage_percentage = min(100, (storage_result['total_size_mb'] / allocated_quota_mb) * 100)
            
            print(f"[📊 사용량] 사용률: {usage_percentage:.1f}% ({storage_result['total_size_mb']:.2f}MB / {allocated_quota_mb}MB)")
            
            return {
                'success': True,
                'storage': {
                    'total_count': storage_result['total_count'],
                    'total_size_bytes': storage_result['total_size_bytes'],
                    'total_size_mb': storage_result['total_size_mb'],
                    'allocated_quota_mb': allocated_quota_mb,
                    'used_mb': storage_result['total_size_mb'],
                    'usage_percentage': round(usage_percentage, 1),
                    'quota_warning_threshold': quota_warning,
                    'remaining_mb': max(0, allocated_quota_mb - storage_result['total_size_mb'])
                },
                'recent_activity': {
                    'thirty_day_count': len(recent_mails),
                    'classification_breakdown': classification_data
                }
            }
        except Exception as e:
            print(f"[❌ 사용량] 사용량 통계 계산 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_daily_mail_stats(user_email, days=7):
        """일별 메일 통계 가져오기"""
        try:
            print(f"[📊 사용량] {user_email} 사용자의 최근 {days}일 일별 통계")
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # 날짜별 메일 수 집계
            daily_stats = {}
            for i in range(days):
                current_date = start_date + timedelta(days=i)
                date_str = current_date.strftime('%Y-%m-%d')
                daily_stats[date_str] = 0
            
            # 기간 내 메일들 조회
            mails = Mail.query.filter(
                and_(
                    Mail.user_email == user_email,
                    Mail.date >= start_date
                )
            ).all()
            
            # 날짜별로 카운트
            for mail in mails:
                date_str = mail.date.strftime('%Y-%m-%d')
                if date_str in daily_stats:
                    daily_stats[date_str] += 1
            
            return {
                'success': True,
                'daily_stats': daily_stats,
                'total_period_mails': len(mails)
            }
        except Exception as e:
            print(f"[❌ 사용량] 일별 통계 계산 실패: {e}")
            return {'success': False, 'error': str(e)}
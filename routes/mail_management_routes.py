"""
메일 관리 라우트 (사용량, 삭제 등)
"""
from flask import Blueprint, request, jsonify, session
from services.usage_service import UsageService
from services.mail_cleanup_service import MailCleanupService

def create_mail_management_routes():
    """메일 관리 라우트 생성"""
    
    mail_mgmt_routes = Blueprint('mail_management', __name__)
    
    # 사용량 관련 라우트
    @mail_mgmt_routes.route('/api/usage/stats', methods=['GET'])
    def get_usage_stats():
        """사용량 통계 가져오기"""
        print("[📊 사용량API] GET /api/usage/stats 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = UsageService.get_usage_statistics(user_email)
        print(f"[📊 사용량API] 통계 조회 결과: success={result['success']}")
        return jsonify(result)
    
    @mail_mgmt_routes.route('/api/usage/daily', methods=['GET'])
    def get_daily_stats():
        """일별 메일 통계 가져오기"""
        print("[📊 사용량API] GET /api/usage/daily 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        days = request.args.get('days', 30, type=int)
        result = UsageService.get_daily_mail_stats(user_email, days)
        print(f"[📊 사용량API] 일별 통계 조회 결과: success={result['success']}")
        return jsonify(result)
    
    # 메일 삭제 관련 라우트
    @mail_mgmt_routes.route('/api/mail-cleanup/settings', methods=['GET'])
    def get_cleanup_settings():
        """메일 삭제 설정 가져오기"""
        print("[🗑️ 정리API] GET /api/mail-cleanup/settings 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = MailCleanupService.get_deletion_settings(user_email)
        print(f"[🗑️ 정리API] 삭제 설정 조회 결과: success={result['success']}")
        return jsonify(result)
    
    @mail_mgmt_routes.route('/api/mail-cleanup/settings', methods=['PUT'])
    def update_cleanup_settings():
        """메일 삭제 설정 업데이트"""
        print("[🗑️ 정리API] PUT /api/mail-cleanup/settings 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '설정 데이터가 필요합니다.'}), 400
        
        result = MailCleanupService.update_deletion_settings(user_email, data)
        print(f"[🗑️ 정리API] 삭제 설정 업데이트 결과: success={result['success']}")
        return jsonify(result)
    
    @mail_mgmt_routes.route('/api/mail-cleanup/preview', methods=['POST'])
    def preview_cleanup():
        """삭제 예상 메일 수 미리보기"""
        print("[🗑️ 정리API] POST /api/mail-cleanup/preview 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = MailCleanupService.preview_cleanup(user_email)
        print(f"[🗑️ 정리API] 삭제 미리보기 결과: success={result['success']}")
        return jsonify(result)
    
    @mail_mgmt_routes.route('/api/mail-cleanup/execute', methods=['POST'])
    def execute_cleanup():
        """메일 자동 삭제 실행"""
        print("[🗑️ 정리API] POST /api/mail-cleanup/execute 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = MailCleanupService.cleanup_old_mails(user_email)
        print(f"[🗑️ 정리API] 메일 삭제 실행 결과: success={result['success']}")
        return jsonify(result)
    
    return mail_mgmt_routes
from flask import Blueprint, request, jsonify
from services.settings_service import SettingsService

def create_settings_routes(session_manager):
    """계층적 설정 관련 라우트 생성"""
    
    settings_routes = Blueprint('settings', __name__)
    settings_service = SettingsService()
    
    # 모든 설정 가져오기
    @settings_routes.route('/api/settings', methods=['GET'])
    def get_all_settings():
        """모든 설정 가져오기"""
        # GET 요청에서는 쿼리 파라미터에서 사용자 이메일 확인
        user_email = request.args.get('email')
        
        if not user_email or not session_manager.session_exists(user_email):
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = settings_service.get_all_settings(user_email)
        return jsonify(result)
    
    # 카테고리별 설정 가져오기
    @settings_routes.route('/api/settings/<category>', methods=['GET'])
    def get_category_settings(category):
        """특정 카테고리의 모든 설정 가져오기"""
        # GET 요청에서는 쿼리 파라미터에서 사용자 이메일 확인
        user_email = request.args.get('email')
        
        if not user_email or not session_manager.session_exists(user_email):
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = settings_service.get_category_settings(user_email, category)
        return jsonify(result)
    
    # 특정 설정 가져오기
    @settings_routes.route('/api/settings/<category>/<subcategory>', methods=['GET'])
    def get_specific_settings(category, subcategory):
        """특정 카테고리/서브카테고리 설정 가져오기"""
        # GET 요청에서는 쿼리 파라미터에서 사용자 이메일 확인
        user_email = request.args.get('email')
        
        if not user_email or not session_manager.session_exists(user_email):
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = settings_service.get_specific_settings(user_email, category, subcategory)
        return jsonify(result)
    
    # 설정 업데이트
    @settings_routes.route('/api/settings/<category>/<subcategory>', methods=['PUT'])
    def update_settings(category, subcategory):
        """설정 업데이트"""
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '데이터가 없습니다.'}), 400
            
        user_email = data.get('email')
        if not user_email or not session_manager.session_exists(user_email):
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = settings_service.update_settings(user_email, category, subcategory, data)
        return jsonify(result)
    
    # 개별 설정값 가져오기
    @settings_routes.route('/api/settings/<category>/<subcategory>/<key>', methods=['GET'])
    def get_setting_value(category, subcategory, key):
        """특정 설정값 하나만 가져오기"""
        data = request.get_json() if request.is_json else {}
        user_email = data.get('email') if data else request.args.get('email')
        
        if not user_email or not session_manager.session_exists(user_email):
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = settings_service.get_setting_value(user_email, category, subcategory, key)
        return jsonify(result)
    
    # 개별 설정값 설정
    @settings_routes.route('/api/settings/<category>/<subcategory>/<key>', methods=['PUT'])
    def set_setting_value(category, subcategory, key):
        """특정 설정값 하나만 설정"""
        data = request.get_json()
        if not data or 'value' not in data:
            return jsonify({'success': False, 'error': 'value가 필요합니다.'}), 400
            
        user_email = data.get('email')
        if not user_email or not session_manager.session_exists(user_email):
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = settings_service.set_setting_value(
            user_email, category, subcategory, key, data['value']
        )
        return jsonify(result)
    
    # 설정 초기화
    @settings_routes.route('/api/settings/reset', methods=['POST'])
    def reset_settings():
        """설정 초기화"""
        data = request.get_json() or {}
        user_email = data.get('email')
        
        if not user_email or not session_manager.session_exists(user_email):
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        data = request.get_json() or {}
        category = data.get('category')
        subcategory = data.get('subcategory')
        
        result = settings_service.reset_settings(user_email, category, subcategory)
        return jsonify(result)
    
    # 설정 구조 정보 제공 (프론트엔드용)
    @settings_routes.route('/api/settings/structure', methods=['GET'])
    def get_settings_structure():
        """설정 구조 정보 제공"""
        print("[🏗️ 설정구조] 설정 구조 정보 요청됨")
        from models.settings_structure import SETTINGS_STRUCTURE
        print(f"[🏗️ 설정구조] 구조 카테고리 수: {len(SETTINGS_STRUCTURE)}")
        return jsonify({'success': True, 'structure': SETTINGS_STRUCTURE})
    
    return settings_routes
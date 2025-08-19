"""
서명 관리 라우트
"""
from flask import Blueprint, request, jsonify, session
from services.signature_service import SignatureService

def create_signature_routes():
    """서명 관리 라우트 생성"""
    
    signature_routes = Blueprint('signature', __name__)
    
    @signature_routes.route('/api/signatures', methods=['GET'])
    def get_signatures():
        """모든 서명 가져오기"""
        print("[🔗 서명API] GET /api/signatures 요청됨")
        user_email = session.get('email')
        print(f"[🔗 서명API] 사용자 이메일: {user_email}")
        if not user_email:
            print("[🔗 서명API] 로그인 필요 - 401 반환")
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = SignatureService.get_signatures(user_email)
        print(f"[🔗 서명API] 서명 조회 결과: success={result['success']}")
        return jsonify(result)
    
    @signature_routes.route('/api/signatures', methods=['POST'])
    def add_or_get_signatures():
        """서명 추가 또는 조회 (WriteMail 호환성)"""
        print("[🔗 서명API] POST /api/signatures 요청됨")
        
        data = request.get_json()
        
        # WriteMail에서 오는 조회 요청 처리 ({email: userEmail})
        if data and 'email' in data and len(data) == 1:
            user_email = data['email']
            print(f"[🔗 서명API] 서명 조회 요청 - 사용자: {user_email}")
            result = SignatureService.get_signatures(user_email)
            print(f"[🔗 서명API] 서명 조회 결과: success={result['success']}")
            return jsonify(result)
        
        # 기존 서명 추가 로직
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        if not data or 'name' not in data or 'content' not in data:
            return jsonify({'success': False, 'error': '서명 이름과 내용이 필요합니다.'}), 400
        
        result = SignatureService.add_signature(
            user_email,
            data['name'],
            data['content'],
            data.get('html_content', ''),
            data.get('is_html', False)
        )
        print(f"[🔗 서명API] 서명 추가 결과: success={result['success']}")
        return jsonify(result)
    
    @signature_routes.route('/api/signatures/<int:signature_id>', methods=['PUT'])
    def update_signature(signature_id):
        """서명 수정"""
        print(f"[🔗 서명API] PUT /api/signatures/{signature_id} 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '수정할 데이터가 없습니다.'}), 400
        
        result = SignatureService.update_signature(
            user_email,
            signature_id,
            data.get('name'),
            data.get('content'),
            data.get('html_content'),
            data.get('is_html')
        )
        print(f"[🔗 서명API] 서명 수정 결과: success={result['success']}")
        return jsonify(result)
    
    @signature_routes.route('/api/signatures/<int:signature_id>', methods=['DELETE'])
    def delete_signature(signature_id):
        """서명 삭제"""
        print(f"[🔗 서명API] DELETE /api/signatures/{signature_id} 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = SignatureService.delete_signature(user_email, signature_id)
        print(f"[🔗 서명API] 서명 삭제 결과: success={result['success']}")
        return jsonify(result)
    
    @signature_routes.route('/api/signatures/status', methods=['GET'])
    def get_signature_status():
        """서명 사용 상태 가져오기"""
        print("[🔗 서명API] GET /api/signatures/status 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = SignatureService.get_signature_status(user_email)
        print(f"[🔗 서명API] 서명 상태 조회 결과: success={result['success']}")
        return jsonify(result)
    
    @signature_routes.route('/api/signatures/status', methods=['PUT'])
    def set_signature_status():
        """서명 사용 상태 설정"""
        print("[🔗 서명API] PUT /api/signatures/status 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        data = request.get_json()
        if not data or 'enabled' not in data:
            return jsonify({'success': False, 'error': '사용 여부가 필요합니다.'}), 400
        
        result = SignatureService.set_signature_status(
            user_email,
            data['enabled'],
            data.get('default_signature')
        )
        print(f"[🔗 서명API] 서명 상태 설정 결과: success={result['success']}")
        return jsonify(result)
    
    @signature_routes.route('/api/signatures/active', methods=['GET'])
    def get_active_signature():
        """현재 활성화된 서명 가져오기"""
        print("[🔗 서명API] GET /api/signatures/active 요청됨")
        user_email = session.get('email')
        if not user_email:
            return jsonify({'success': False, 'error': '로그인이 필요합니다.'}), 401
        
        result = SignatureService.get_active_signature(user_email)
        print(f"[🔗 서명API] 활성 서명 조회 결과: success={result['success']}")
        return jsonify(result)
    
    return signature_routes
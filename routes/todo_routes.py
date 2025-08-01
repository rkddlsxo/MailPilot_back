from flask import Blueprint, request, jsonify
from datetime import datetime

def create_todo_routes(session_manager, todo_service):
    todo_bp = Blueprint('todo', __name__)
    
    @todo_bp.route('/api/todos', methods=['GET', 'POST', 'PUT', 'DELETE'])
    def manage_todos():
        """할일 관리 API"""
        try:
            if request.method == 'GET':
                user_email = request.args.get('email')
            else:
                user_email = request.json.get('email') if request.json else None
                
            if not user_email:
                return jsonify({"error": "이메일이 필요합니다."}), 400
            
            if not session_manager.session_exists(user_email):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            user_session = session_manager.get_session(user_email)
            
            if request.method == 'GET':
                # 할일 목록 조회
                todos = user_session.get('extracted_todos', [])
                return jsonify({
                    "success": True,
                    "todos": todos,
                    "total_count": len(todos)
                })
            
            elif request.method == 'POST':
                # 새 할일 추가
                data = request.json
                
                existing_todos = user_session.get('extracted_todos', [])
                max_id = max([todo.get('id', 0) for todo in existing_todos] + [0])
                new_id = max_id + 1
                
                new_todo = {
                    'id': new_id,
                    'type': data.get('type', 'task'),
                    'title': data.get('title', ''),
                    'description': data.get('description', ''),
                    'date': data.get('date'),
                    'time': data.get('time'),
                    'priority': data.get('priority', 'medium'),
                    'status': 'pending',
                    'editable_date': True,
                    'source_email': {
                        'from': 'manual',
                        'subject': 'Manual Entry',
                        'date': datetime.now().isoformat(),
                        'type': 'manual_entry'
                    }
                }
                
                existing_todos.append(new_todo)
                user_session['extracted_todos'] = existing_todos
                
                # 파일에 자동 저장
                session_manager.save_user_session_to_file(user_email)
                
                return jsonify({
                    "success": True,
                    "todo": new_todo,
                    "message": "할일이 추가되고 저장되었습니다."
                })
            
            elif request.method == 'PUT':
                # 할일 업데이트
                data = request.json
                todo_id = data.get('id')
                
                todos = user_session.get('extracted_todos', [])
                updated = False
                
                for todo in todos:
                    if todo.get('id') == todo_id:
                        if 'status' in data:
                            todo['status'] = data['status']
                        if 'date' in data and todo.get('editable_date', True):
                            todo['date'] = data['date']
                        if 'time' in data and todo.get('editable_date', True):
                            todo['time'] = data['time']
                        
                        updated = True
                        break
                
                if updated:
                    user_session['extracted_todos'] = todos
                    session_manager.save_user_session_to_file(user_email)
                    
                    return jsonify({
                        "success": True,
                        "message": "할일이 업데이트되고 저장되었습니다."
                    })
                else:
                    return jsonify({"error": "해당 할일을 찾을 수 없습니다."}), 404
            
            elif request.method == 'DELETE':
                # 할일 삭제
                data = request.json
                todo_id = data.get('id')
                
                todos = user_session.get('extracted_todos', [])
                original_count = len(todos)
                
                todos = [todo for todo in todos if todo.get('id') != todo_id]
                
                if len(todos) < original_count:
                    user_session['extracted_todos'] = todos
                    session_manager.save_user_session_to_file(user_email)
                    
                    return jsonify({
                        "success": True,
                        "message": "할일이 삭제되고 저장되었습니다."
                    })
                else:
                    return jsonify({"error": "해당 할일을 찾을 수 없습니다."}), 404
            
        except Exception as e:
            print(f"[❗할일 API 오류] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @todo_bp.route('/api/extract-todos', methods=['POST'])
    def extract_todos():
        """이메일에서 할일 추출"""
        try:
            data = request.get_json()
            user_email = data.get("email", "")
            email_ids = data.get("email_ids", [])
            
            print(f"[📋 할일 추출] 사용자: {user_email}")
            
            if not session_manager.session_exists(user_email):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            user_session = session_manager.get_session(user_email)
            last_emails = user_session.get('last_emails', [])
            
            all_todos = []
            processed_count = 0
            
            emails_to_process = last_emails
            if email_ids:
                emails_to_process = [email for email in last_emails if email.get('id') in email_ids]
            
            for email_data in emails_to_process:
                try:
                    result = todo_service.extract_todos_from_email(
                        email_body=email_data.get('body', ''),
                        email_subject=email_data.get('subject', ''),
                        email_from=email_data.get('from', ''),
                        email_date=email_data.get('date', '')
                    )
                    
                    if result['success']:
                        for todo in result['todos']:
                            todo['source_email']['id'] = email_data.get('id')
                            todo['source_email']['subject'] = email_data.get('subject', '')
                        
                        all_todos.extend(result['todos'])
                        processed_count += 1
                        
                except Exception as e:
                    continue
            
            # 기존 할일과 병합
            existing_todos = user_session.get('extracted_todos', [])
            existing_ids = {todo.get('id') for todo in existing_todos}
            
            new_todos = [todo for todo in all_todos if todo.get('id') not in existing_ids]
            final_todos = existing_todos + new_todos
            
            final_todos.sort(key=lambda x: x['date'] or '9999-12-31')
            
            user_session['extracted_todos'] = final_todos
            
            # 파일에 자동 저장
            session_manager.save_user_session_to_file(user_email)
            
            print(f"[✅ 할일 추출 + 저장] 총 {len(final_todos)}개 (신규 {len(new_todos)}개)")
            
            return jsonify({
                "success": True,
                "todos": final_todos,
                "total_count": len(final_todos),
                "new_todos": len(new_todos),
                "processed_emails": processed_count
            })
            
        except Exception as e:
            print(f"[❗할일 추출 오류] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return todo_bp
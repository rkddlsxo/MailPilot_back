from flask import Blueprint, request, jsonify
from datetime import datetime
from models.tables import db, Todo, Mail

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
            
            if request.method == 'GET':
                # DB에서 할일 목록 조회
                db_todos = Todo.query.filter_by(user_email=user_email).all()
                todos = [{
                    'id': todo.todo_id,
                    'title': todo.title,
                    'type': todo.type,
                    'description': todo.event or '',
                    'date': todo.date.strftime('%Y-%m-%d') if todo.date else None,
                    'time': todo.time,
                    'priority': todo.priority,
                    'status': todo.status,
                    'mail_id': todo.mail_id
                } for todo in db_todos]
                
                return jsonify({
                    "success": True,
                    "todos": todos,
                    "total_count": len(todos)
                })
            
            elif request.method == 'POST':
                # DB에 새 할일 추가
                data = request.json
                
                # 중복 검사
                title = data.get('title', '').strip()
                todo_type = data.get('type', 'task')
                
                existing_todo = Todo.query.filter_by(
                    user_email=user_email,
                    title=title,
                    type=todo_type
                ).first()
                
                if existing_todo:
                    return jsonify({
                        "success": False,
                        "error": "중복된 할일입니다."
                    }), 409
                
                # 날짜 변환
                todo_date = None
                if data.get('date'):
                    try:
                        todo_date = datetime.strptime(data.get('date'), '%Y-%m-%d').date()
                    except:
                        pass
                
                # DB에 새 할일 저장
                new_todo = Todo(
                    user_email=user_email,
                    title=title,
                    type=todo_type,
                    event=data.get('description', ''),
                    date=todo_date,
                    time=data.get('time'),
                    priority=data.get('priority', 'medium'),
                    status='pending',
                    mail_id=data.get('mail_id')  # 수동 추가시 null
                )
                
                db.session.add(new_todo)
                db.session.commit()
                
                return jsonify({
                    "success": True,
                    "todo": {
                        'id': new_todo.todo_id,
                        'title': new_todo.title,
                        'type': new_todo.type,
                        'description': new_todo.event or '',
                        'date': new_todo.date.strftime('%Y-%m-%d') if new_todo.date else None,
                        'time': new_todo.time,
                        'priority': new_todo.priority,
                        'status': new_todo.status,
                        'mail_id': new_todo.mail_id
                    },
                    "message": "할일이 추가되었습니다."
                })
            
            elif request.method == 'PUT':
                # DB에서 할일 업데이트
                data = request.json
                todo_id = data.get('id')
                
                todo = Todo.query.filter_by(todo_id=todo_id, user_email=user_email).first()
                
                if not todo:
                    return jsonify({"error": "해당 할일을 찾을 수 없습니다."}), 404
                
                # 업데이트 가능한 필드들
                if 'status' in data:
                    todo.status = data['status']
                if 'date' in data:
                    try:
                        todo.date = datetime.strptime(data['date'], '%Y-%m-%d').date() if data['date'] else None
                    except:
                        pass
                if 'time' in data:
                    todo.time = data['time']
                if 'priority' in data:
                    todo.priority = data['priority']
                
                db.session.commit()
                
                return jsonify({
                    "success": True,
                    "message": "할일이 업데이트되었습니다."
                })
            
            elif request.method == 'DELETE':
                # DB에서 할일 삭제
                data = request.json
                todo_id = data.get('id')
                
                todo = Todo.query.filter_by(todo_id=todo_id, user_email=user_email).first()
                
                if not todo:
                    return jsonify({"error": "해당 할일을 찾을 수 없습니다."}), 404
                
                db.session.delete(todo)
                db.session.commit()
                
                return jsonify({
                    "success": True,
                    "message": "할일이 삭제되었습니다."
                })
            
        except Exception as e:
            print(f"[❌ 할일] {str(e)}")
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
            
            # 1. DB에서 메일 데이터 조회 (세션 대신 DB 사용)
            if email_ids:
                # 특정 메일들만 처리
                db_mails = Mail.query.filter(
                    Mail.user_email == user_email,
                    Mail.mail_id.in_(email_ids)
                ).all()
            else:
                # 모든 메일 처리
                db_mails = Mail.query.filter_by(user_email=user_email).all()
            
            emails_to_process = [{
                'id': mail.mail_id,
                'subject': mail.subject,
                'body': mail.body,
                'from': mail.from_,
                'date': mail.date.strftime('%Y-%m-%d %H:%M:%S') if mail.date else ''
            } for mail in db_mails]
            
            # 2. DB에서 기존 할일 조회
            existing_todos = Todo.query.filter_by(user_email=user_email).all()
            existing_keys = {f"{todo.title.lower().strip()}_{todo.type}" for todo in existing_todos}
            
            all_todos = []
            processed_count = 0
            new_count = 0
            
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
                            # 2. 중복 검사
                            todo_key = f"{todo['title'].lower().strip()}_{todo['type']}"
                            
                            if todo_key not in existing_keys:
                                # 3. 중복 아니면 DB 저장
                                todo_date = None
                                if todo.get('date'):
                                    try:
                                        todo_date = datetime.strptime(todo['date'], '%Y-%m-%d').date()
                                    except:
                                        pass
                                
                                new_todo = Todo(
                                    user_email=user_email,
                                    title=todo['title'],
                                    type=todo['type'],
                                    event=todo.get('description', ''),
                                    date=todo_date,
                                    time=todo.get('time'),
                                    priority=todo.get('priority', 'medium'),
                                    status='pending',
                                    mail_id=email_data.get('id')
                                )
                                
                                db.session.add(new_todo)
                                existing_keys.add(todo_key)  # 메모리에서도 중복 방지
                                
                                # 응답용 데이터 추가
                                all_todos.append({
                                    'title': todo['title'],
                                    'type': todo['type'],
                                    'description': todo.get('description', ''),
                                    'date': todo.get('date'),
                                    'time': todo.get('time'),
                                    'priority': todo.get('priority', 'medium'),
                                    'status': 'pending',
                                    'mail_id': email_data.get('id')
                                })
                                new_count += 1
                        
                        processed_count += 1
                        
                except Exception as e:
                    print(f"[❌ 할일추출] {str(e)}")
                    continue
            
            # DB에 커밋
            db.session.commit()
            
            # 최종 할일 목록 조회
            final_todos = Todo.query.filter_by(user_email=user_email).all()
            todos_response = [{
                'id': todo.todo_id,
                'title': todo.title,
                'type': todo.type,
                'description': todo.event or '',
                'date': todo.date.strftime('%Y-%m-%d') if todo.date else None,
                'time': todo.time,
                'priority': todo.priority,
                'status': todo.status,
                'mail_id': todo.mail_id
            } for todo in final_todos]
            
            print(f"[✅ 할일추출] 총 {len(todos_response)}개 (신규 {new_count}개)")
            
            return jsonify({
                "success": True,
                "todos": todos_response,
                "total_count": len(todos_response),
                "new_todos": new_count,
                "processed_emails": processed_count
            })
            
        except Exception as e:
            print(f"[❌ 할일추출] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    @todo_bp.route('/api/todos/cleanup-duplicates', methods=['POST'])
    def cleanup_duplicates():
        """중복 할일 정리"""
        try:
            data = request.get_json()
            user_email = data.get("email", "")
            
            if not user_email:
                return jsonify({"error": "이메일이 필요합니다."}), 400
            
            if not session_manager.session_exists(user_email):
                return jsonify({"error": "로그인이 필요합니다."}), 401
            
            # DB에서 사용자의 모든 할일 조회
            todos = Todo.query.filter_by(user_email=user_email).all()
            original_count = len(todos)
            
            print(f"[🔄 중복 정리 시작] {original_count}개 할일")
            
            # 중복 제거 로직
            seen_todos = set()
            todos_to_keep = []
            todos_to_remove = []
            
            for todo in todos:
                todo_key = f"{todo.title.lower().strip()}_{todo.type}"
                
                if todo_key not in seen_todos:
                    seen_todos.add(todo_key)
                    todos_to_keep.append(todo)
                else:
                    todos_to_remove.append(todo)
                    print(f"[🗑️ 중복 제거] {todo.title} ({todo.type})")
            
            # 중복 할일들 DB에서 삭제
            for todo in todos_to_remove:
                db.session.delete(todo)
            
            db.session.commit()
            
            removed_count = len(todos_to_remove)
            remaining_count = len(todos_to_keep)
            
            print(f"[✅ 중복정리] {removed_count}개 제거, {remaining_count}개 남음")
            
            return jsonify({
                "success": True,
                "message": f"{removed_count}개의 중복 할일이 제거되었습니다.",
                "removed_count": removed_count,
                "remaining_count": remaining_count,
                "original_count": original_count
            })
            
        except Exception as e:
            print(f"[❌ 중복정리] {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return todo_bp
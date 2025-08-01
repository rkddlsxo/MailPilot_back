# services/todo_service.py - 할일 추출 서비스

import re
import time
from datetime import datetime, timedelta

# 선택적 임포트
try:
    import dateutil.parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    print("[⚠️ python-dateutil 없음 - 고급 날짜 파싱 비활성화]")

class TodoService:
    def __init__(self, config):
        self.config = config
        
        # 할일 추출을 위한 키워드 패턴들
        self.todo_keywords = {
            'meeting': ['회의', '미팅', 'meeting', '컨퍼런스', '세미나', '면담', '상담'],
            'deadline': ['마감', '제출', '완료', '끝내', 'deadline', 'due', '기한', '까지'],
            'task': ['작업', '업무', '처리', '진행', '해야', '할것', 'task', 'work', 'todo'],
            'event': ['행사', '이벤트', 'event', '파티', '모임', '약속', '일정'],
            'reminder': ['알림', 'reminder', '잊지말', '기억', '체크', '확인']
        }
        
        # 날짜/시간 추출 패턴
        self.date_patterns = [
            r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
            r'(\d{1,2})월\s*(\d{1,2})일',
            r'(\d{1,2})/(\d{1,2})',
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(오늘|내일|모레)',
            r'(다음주|이번주|다다음주)',
            r'(월요일|화요일|수요일|목요일|금요일|토요일|일요일)'
        ]
        
        self.time_patterns = [
            r'(\d{1,2}):(\d{2})',
            r'(\d{1,2})시\s*(\d{1,2})?분?',
            r'(오전|오후)\s*(\d{1,2})시',
        ]
        
        print("[📋 할일 서비스 초기화 완료]")
    
    def extract_todos_from_email(self, email_body, email_subject, email_from, email_date):
        """이메일에서 할일 추출"""
        try:
            print(f"[📋 할일 추출] {email_subject[:30]}...")
            
            full_text = f"{email_subject} {email_body}"
            todos = []
            
            # 고유 ID 생성을 위한 기준 시간
            base_timestamp = int(time.time() * 1000)
            
            # 1. 회의/미팅 추출
            try:
                meeting_todos = self._extract_meetings(full_text, email_from, email_date, email_subject, base_timestamp)
                todos.extend(meeting_todos)
            except Exception as e:
                print(f"[⚠️ 회의 추출 오류] {str(e)}")

            # 2. 마감일/데드라인 추출  
            deadline_todos = self._extract_deadlines(full_text, email_from, email_date, email_subject, base_timestamp + 100)
            todos.extend(deadline_todos)
            
            # 3. 일반 할일 추출
            task_todos = self._extract_general_tasks(full_text, email_from, email_date, email_subject, base_timestamp + 200)
            todos.extend(task_todos)
            
            # 4. 이벤트/행사 추출
            event_todos = self._extract_events(full_text, email_from, email_date, email_subject, base_timestamp + 300)
            todos.extend(event_todos)
            
            # 중복 제거 및 우선순위 설정
            todos = self._deduplicate_todos(todos)
            todos = self._assign_priority(todos)
            
            print(f"[✅ 할일 추출 완료] {len(todos)}개 발견")
            
            return {
                'success': True,
                'todos': todos,
                'total_count': len(todos),
                'extraction_method': 'improved_ai_analysis'
            }
            
        except Exception as e:
            print(f"[❗할일 추출 오류] {str(e)}")
            return {
                'success': False,
                'todos': [],
                'error': str(e)
            }
    
    def _extract_meetings(self, text, sender, email_date, email_subject, base_id):
        """회의/미팅 추출"""
        meetings = []
        
        for keyword in self.todo_keywords['meeting']:
            if keyword.lower() in text.lower():
                meeting_title = self._generate_smart_title(text, keyword, email_subject, 'meeting')
                meeting_date = self._extract_smart_date(text) or '2024-12-27'
                meeting_time = self._extract_smart_time(text) or '14:00'
                
                meeting = {
                    'id': base_id + len(meetings),
                    'type': 'meeting',
                    'title': meeting_title,
                    'description': f"{sender}님과의 {keyword}",
                    'date': meeting_date,
                    'time': meeting_time,
                    'priority': 'high',
                    'status': 'pending',
                    'editable_date': True,
                    'source_email': {
                        'from': sender,
                        'subject': email_subject,
                        'date': email_date,
                        'type': 'meeting_invitation'
                    }
                }
                meetings.append(meeting)
                print(f"[🤝 회의 추출] {meeting_title} (ID: {meeting['id']})")
                break
        
        return meetings
    
    def _extract_deadlines(self, text, sender, email_date, email_subject, base_id):
        """마감일 추출"""
        deadlines = []
        
        for keyword in self.todo_keywords['deadline']:
            if keyword.lower() in text.lower():
                deadline_title = self._generate_smart_title(text, keyword, email_subject, 'deadline')
                deadline_date = self._extract_smart_date(text) or '2024-12-28'
                
                deadline = {
                    'id': base_id + len(deadlines),
                    'type': 'deadline',
                    'title': deadline_title,
                    'description': f"{sender}님이 요청한 마감 업무",
                    'date': deadline_date,
                    'time': None,
                    'priority': 'high',
                    'status': 'pending',
                    'editable_date': True,
                    'source_email': {
                        'from': sender,
                        'subject': email_subject,
                        'date': email_date,
                        'type': 'deadline_notice'
                    }
                }
                deadlines.append(deadline)
                print(f"[⏰ 마감일 추출] {deadline_title} (ID: {deadline['id']})")
                break
        
        return deadlines
    
    def _extract_general_tasks(self, text, sender, email_date, email_subject, base_id):
        """일반 업무 추출"""
        tasks = []
        
        task_patterns = [
            (r'([^\n\.]{10,80})\s*(해주세요|해주시기|부탁드립니다|요청드립니다)', 'korean_request'),
            (r'([^\n\.]{10,80})\s*(확인|검토|처리|진행)\s*(해주세요|부탁|필요)', 'korean_action'),
            (r'(please|kindly)\s+([^\n\.]{10,80})', 'english_request'),
            (r'([^\n\.]{10,80})\s+(please|kindly)', 'english_request_after'),
            (r'(could you|can you|would you)\s+([^\n\.]{10,80})', 'english_question'),
        ]
        
        for pattern, pattern_type in task_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if pattern_type == 'english_request':
                    task_name = match.group(2).strip()
                elif pattern_type == 'english_request_after':
                    task_name = match.group(1).strip()
                elif pattern_type == 'english_question':
                    task_name = match.group(2).strip()
                else:
                    task_name = match.group(1).strip()
                
                if len(task_name) > 10 and not self._is_meaningless_text(task_name):
                    clean_title = self._clean_task_title(task_name)
                    
                    task = {
                        'id': base_id + len(tasks),
                        'type': 'task',
                        'title': clean_title,
                        'description': f"{sender}님이 요청한 업무",
                        'date': None,
                        'time': None,
                        'priority': 'medium',
                        'status': 'pending',
                        'editable_date': True,
                        'source_email': {
                            'from': sender,
                            'subject': email_subject,
                            'date': email_date,
                            'type': 'task_request'
                        }
                    }
                    tasks.append(task)
                    print(f"[📋 업무 추출] {clean_title} (ID: {task['id']})")
                    
                    if len(tasks) >= 3:
                        break
        
        return tasks
    
    def _extract_events(self, text, sender, email_date, email_subject, base_id):
        """이벤트 추출"""
        events = []
        
        for keyword in self.todo_keywords['event']:
            if keyword.lower() in text.lower():
                event_title = self._generate_smart_title(text, keyword, email_subject, 'event')
                
                event = {
                    'id': base_id + len(events),
                    'type': 'event',
                    'title': event_title,
                    'description': f"{sender}님이 알린 {keyword}",
                    'date': self._extract_smart_date(text) or '2024-12-29',
                    'time': self._extract_smart_time(text) or '18:00',
                    'priority': 'medium',
                    'status': 'pending',
                    'editable_date': True,
                    'source_email': {
                        'from': sender,
                        'subject': email_subject,
                        'date': email_date,
                        'type': 'event_notification'
                    }
                }
                events.append(event)
                print(f"[🎉 이벤트 추출] {event_title} (ID: {event['id']})")
                break
        
        return events
    
    def _generate_smart_title(self, text, keyword, email_subject, todo_type):
        """스마트한 제목 생성"""
        # 1. 이메일 제목에 키워드가 있으면 제목 사용
        if keyword.lower() in email_subject.lower():
            return email_subject[:60]
        
        # 2. 본문에서 키워드 주변 문장 찾기
        sentences = text.split('.')
        for sentence in sentences:
            if keyword.lower() in sentence.lower() and len(sentence.strip()) > 10:
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 60:
                    clean_sentence = clean_sentence[:60] + "..."
                return clean_sentence
        
        # 3. 기본 제목 생성
        type_names = {
            'meeting': '회의',
            'deadline': '마감일',
            'task': '업무',
            'event': '이벤트'
        }
        
        base_name = type_names.get(todo_type, '할일')
        return f"{base_name}: {email_subject[:40]}"
    
    def _extract_smart_date(self, text):
        """스마트 날짜 추출"""
        # 한국어 날짜 패턴들
        for pattern in self.date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if len(match.groups()) == 3:  # 년월일
                        year, month, day = match.groups()
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    elif len(match.groups()) == 2:  # 월일
                        month, day = match.groups()
                        year = datetime.now().year
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except:
                    continue
        
        # 상대적 날짜
        today = datetime.now()
        if '오늘' in text:
            return today.strftime('%Y-%m-%d')
        elif '내일' in text:
            return (today + timedelta(days=1)).strftime('%Y-%m-%d')
        elif '다음주' in text:
            return (today + timedelta(days=7)).strftime('%Y-%m-%d')
        
        return None
    
    def _extract_smart_time(self, text):
        """스마트 시간 추출"""
        for pattern in self.time_patterns:
            match = re.search(pattern, text)
            if match:
                if '오전' in pattern:
                    hour = int(match.group(1))
                    return f"{hour:02d}:00"
                elif '오후' in pattern:
                    hour = int(match.group(1))
                    if hour != 12:
                        hour += 12
                    return f"{hour:02d}:00"
                elif ':' in match.group(0):
                    return match.group(0)
                else:
                    hour = int(match.group(1))
                    minute = match.group(2) if len(match.groups()) > 1 and match.group(2) else "00"
                    return f"{hour:02d}:{minute.zfill(2)}"
        
        return None
    
    def _is_meaningless_text(self, text):
        """의미없는 텍스트인지 확인"""
        meaningless_patterns = [
            r'^[^a-zA-Z가-힣]*$',  # 문자가 없음
            r'^(please|kindly|확인|검토|처리)$',  # 단일 키워드만
            r'^.{1,5}$',  # 너무 짧음
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        
        return False
    
    def _clean_task_title(self, title):
        """할일 제목 정리"""
        remove_words = ['해주세요', '부탁드립니다', '요청드립니다', 'please', 'kindly']
        
        clean_title = title
        for word in remove_words:
            clean_title = clean_title.replace(word, '').strip()
        
        # 첫 글자 대문자 처리 (영어인 경우)
        if clean_title and clean_title[0].isalpha():
            clean_title = clean_title[0].upper() + clean_title[1:]
        
        # 길이 제한
        if len(clean_title) > 60:
            clean_title = clean_title[:60] + "..."
        
        return clean_title
    
    def _deduplicate_todos(self, todos):
        """중복 제거"""
        seen_titles = set()
        unique_todos = []
        
        for todo in todos:
            title_key = todo['title'].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_todos.append(todo)
            else:
                print(f"[🗑️ 중복 제거] {todo['title']}")
        
        return unique_todos
    
    def _assign_priority(self, todos):
        """우선순위 자동 설정"""
        for todo in todos:
            # 키워드 기반 우선순위
            if todo['type'] == 'deadline':
                todo['priority'] = 'high'
            elif todo['type'] == 'meeting':
                todo['priority'] = 'high'
            elif '긴급' in todo['title'] or 'urgent' in todo['title'].lower():
                todo['priority'] = 'high'
            elif todo['type'] == 'event':
                todo['priority'] = 'medium'
            else:
                todo['priority'] = 'low'
            
            # 날짜 기반 우선순위 조정
            if todo['date']:
                try:
                    todo_date = datetime.fromisoformat(todo['date'].replace('Z', '+00:00')).replace(tzinfo=None)
                    days_until = (todo_date - datetime.now()).days
                    
                    if days_until <= 1:  # 오늘/내일
                        todo['priority'] = 'high'
                    elif days_until <= 3:  # 3일 이내
                        if todo['priority'] == 'low':
                            todo['priority'] = 'medium'
                except:
                    pass
        
        return todos
    
    def extract_dates_from_text(self, text):
        """텍스트에서 날짜 추출"""
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    date_str = match.group(0)
                    parsed_date = self._parse_korean_date(date_str)
                    
                    if parsed_date:
                        dates.append({
                            'original_text': date_str,
                            'parsed_date': parsed_date.isoformat(),
                            'confidence': 0.8
                        })
                except Exception:
                    continue
        
        return dates
    
    def extract_times_from_text(self, text):
        """텍스트에서 시간 추출"""  
        times = []
        
        for pattern in self.time_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    time_str = match.group(0)
                    parsed_time = self._parse_korean_time(time_str)
                    
                    if parsed_time:
                        times.append({
                            'original_text': time_str,
                            'parsed_time': parsed_time,
                            'confidence': 0.8
                        })
                except Exception:
                    continue
        
        return times
    
    def _parse_korean_date(self, date_str):
        """한국어 날짜 문자열을 datetime으로 변환"""
        try:
            # 상대적 날짜 처리
            today = datetime.now()
            
            if '오늘' in date_str:
                return today
            elif '내일' in date_str:
                return today + timedelta(days=1)
            elif '모레' in date_str:
                return today + timedelta(days=2)
            elif '다음주' in date_str:
                return today + timedelta(days=7)
            
            # 숫자 날짜 처리
            korean_date_match = re.search(r'(\d{4})?년?\s*(\d{1,2})월\s*(\d{1,2})일', date_str)
            if korean_date_match:
                year = korean_date_match.group(1) or today.year
                month = int(korean_date_match.group(2))
                day = int(korean_date_match.group(3))
                return datetime(int(year), month, day)
            
            # dateutil 사용 (있는 경우)
            if DATEUTIL_AVAILABLE:
                return dateutil.parser.parse(date_str, fuzzy=True)
            
            return None
            
        except Exception as e:
            return None
    
    def _parse_korean_time(self, time_str):
        """한국어 시간 문자열 파싱"""
        try:
            # 오전/오후 처리
            if '오전' in time_str:
                hour_match = re.search(r'(\d{1,2})시', time_str)
                if hour_match:
                    hour = int(hour_match.group(1))
                    return f"{hour:02d}:00"
            
            elif '오후' in time_str:
                hour_match = re.search(r'(\d{1,2})시', time_str)
                if hour_match:
                    hour = int(hour_match.group(1))
                    if hour != 12:
                        hour += 12
                    return f"{hour:02d}:00"
            
            # 24시간 형식
            time_match = re.search(r'(\d{1,2}):(\d{2})', time_str)
            if time_match:
                return f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
            
            return None
            
        except Exception as e:
            return None
    
    def get_todo_stats(self, todos):
        """할일 통계 생성"""
        stats = {
            'total': len(todos),
            'by_type': {},
            'by_priority': {},
            'by_status': {},
            'upcoming': 0,
            'overdue': 0
        }
        
        today = datetime.now().date()
        
        for todo in todos:
            # 타입별 통계
            todo_type = todo.get('type', 'unknown')
            stats['by_type'][todo_type] = stats['by_type'].get(todo_type, 0) + 1
            
            # 우선순위별 통계
            priority = todo.get('priority', 'low')
            stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1
            
            # 상태별 통계
            status = todo.get('status', 'pending')
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # 날짜 기반 통계
            if todo.get('date'):
                try:
                    todo_date = datetime.fromisoformat(todo['date']).date()
                    if todo_date >= today:
                        stats['upcoming'] += 1
                    else:
                        stats['overdue'] += 1
                except:
                    pass
        
        return stats
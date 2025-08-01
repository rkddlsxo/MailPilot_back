import re
from datetime import datetime, timedelta
import dateutil.parser

class DateUtils:
    # 날짜 패턴들
    DATE_PATTERNS = [
        r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',  # 2024년 12월 25일
        r'(\d{1,2})월\s*(\d{1,2})일',              # 12월 25일
        r'(\d{1,2})/(\d{1,2})',                    # 12/25
        r'(\d{4}-\d{1,2}-\d{1,2})',               # 2024-12-25
        r'(오늘|내일|모레)',                        # 상대적 날짜
        r'(다음주|이번주|다다음주)',                 # 상대적 주
        r'(월요일|화요일|수요일|목요일|금요일|토요일|일요일)'  # 요일
    ]
    
    TIME_PATTERNS = [
        r'(\d{1,2}):(\d{2})',                      # 14:30
        r'(\d{1,2})시\s*(\d{1,2})?분?',           # 2시 30분
        r'(오전|오후)\s*(\d{1,2})시',              # 오전 10시
    ]
    
    @classmethod
    def parse_korean_date(cls, date_str):
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
            
            # 다른 형식들도 시도
            return dateutil.parser.parse(date_str, fuzzy=True)
            
        except Exception as e:
            return None
    
    @classmethod
    def parse_korean_time(cls, time_str):
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
    
    @classmethod
    def extract_dates_from_text(cls, text):
        """텍스트에서 날짜 추출"""
        dates = []
        
        for pattern in cls.DATE_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    date_str = match.group(0)
                    parsed_date = cls.parse_korean_date(date_str)
                    
                    if parsed_date:
                        dates.append({
                            'original_text': date_str,
                            'parsed_date': parsed_date.isoformat(),
                            'confidence': 0.8
                        })
                except Exception:
                    continue
        
        return dates
    
    @classmethod
    def extract_times_from_text(cls, text):
        """텍스트에서 시간 추출"""  
        times = []
        
        for pattern in cls.TIME_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    time_str = match.group(0)
                    parsed_time = cls.parse_korean_time(time_str)
                    
                    if parsed_time:
                        times.append({
                            'original_text': time_str,
                            'parsed_time': parsed_time,
                            'confidence': 0.8
                        })
                except Exception:
                    continue
        
        return times
    
    @staticmethod
    def format_date_korean(date_obj):
        """datetime을 한국어 형식으로 포매팅"""
        if not date_obj:
            return ""
        
        weekdays = ['월', '화', '수', '목', '금', '토', '일']
        weekday = weekdays[date_obj.weekday()]
        
        return f"{date_obj.year}년 {date_obj.month}월 {date_obj.day}일 ({weekday})"
    
    @staticmethod
    def days_until(target_date):
        """목표 날짜까지 남은 일수 계산"""
        if isinstance(target_date, str):
            try:
                target_date = datetime.fromisoformat(target_date.replace('Z', '+00:00')).replace(tzinfo=None)
            except:
                return None
        
        if not isinstance(target_date, datetime):
            return None
        
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        target = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return (target - today).days

import re
from email.header import decode_header

class TextUtils:
    @staticmethod
    def decode_email_header(raw_header):
        """이메일 헤더 디코딩"""
        try:
            decoded_parts = decode_header(raw_header)
            if decoded_parts and decoded_parts[0]:
                decoded_header = decoded_parts[0]
                if isinstance(decoded_header[0], bytes):
                    encoding = decoded_header[1] or 'utf-8'
                    try:
                        return decoded_header[0].decode(encoding)
                    except (UnicodeDecodeError, LookupError):
                        # 여러 인코딩으로 시도
                        for fallback_encoding in ['utf-8', 'latin-1', 'cp949', 'euc-kr']:
                            try:
                                return decoded_header[0].decode(fallback_encoding)
                            except (UnicodeDecodeError, LookupError):
                                continue
                        return decoded_header[0].decode('utf-8', errors='ignore')
                else:
                    return str(decoded_header[0])
            return "(제목 없음)"
        except Exception:
            return raw_header if raw_header else "(제목 없음)"
    
    @staticmethod
    def clean_text(text, max_length=None):
        """텍스트 정리"""
        if not text:
            return ""
        
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 길이 제한
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    @staticmethod
    def extract_email_addresses(text):
        """텍스트에서 이메일 주소 추출"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    @staticmethod
    def is_korean_text(text):
        """한국어 텍스트인지 확인"""
        korean_chars = re.findall(r'[가-힣]', text)
        return len(korean_chars) > 0
    
    @staticmethod
    def truncate_text(text, max_length, suffix="..."):
        """텍스트 자르기"""
        if len(text) <= max_length:
            return text
        return text[:max_length-len(suffix)] + suffix
import os
import mimetypes
from pathlib import Path

class FileUtils:
    # 지원되는 파일 확장자
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls'}
    TEXT_EXTENSIONS = {'.txt', '.md', '.csv'}
    
    @staticmethod
    def get_file_type(filename):
        """파일 확장자로 파일 타입 결정"""
        if not filename:
            return 'unknown'
        
        ext = Path(filename).suffix.lower()
        
        if ext in FileUtils.IMAGE_EXTENSIONS:
            return 'image'
        elif ext in FileUtils.DOCUMENT_EXTENSIONS:
            return 'document'
        elif ext in FileUtils.TEXT_EXTENSIONS:
            return 'text'
        else:
            return 'other'
    
    @staticmethod
    def get_mime_type(filename):
        """파일명으로 MIME 타입 추정"""
        if not filename:
            return 'application/octet-stream'
        
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'
    
    @staticmethod
    def format_file_size(size_bytes):
        """파일 크기를 사람이 읽기 쉬운 형태로 변환"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        if i == 0:
            return f"{int(size_bytes)}{size_names[i]}"
        else:
            return f"{size_bytes:.1f}{size_names[i]}"
    
    @staticmethod
    def is_safe_filename(filename):
        """안전한 파일명인지 확인"""
        if not filename:
            return False
        
        # 위험한 문자들 확인
        dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            if char in filename:
                return False
        
        # 길이 확인
        if len(filename) > 255:
            return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename):
        """파일명 안전하게 변환"""
        if not filename:
            return "unnamed_file"
        
        # 위험한 문자들 제거
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # 연속된 언더스코어 제거
        safe_filename = re.sub(r'_+', '_', safe_filename)
        
        # 길이 제한
        if len(safe_filename) > 200:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:200 - len(ext)] + ext
        
        return safe_filename
    
    @staticmethod
    def ensure_directory(directory_path):
        """디렉토리 존재 확인 및 생성"""
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"[❗디렉토리 생성 실패] {directory_path}: {str(e)}")
            return False
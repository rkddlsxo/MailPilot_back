import imaplib
import smtplib
import email as email_module
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta
from models.db import db
from models.tables import Mail

class EmailService:
    def __init__(self, config, summarizer=None, ai_models=None):
        self.config = config
        self.summarizer = summarizer
        self.ai_models = ai_models
    
    def connect_imap(self, username, password):
        """IMAP 연결"""
        try:
            print(f"[🔍 디버그] IMAP 서버 연결 중: {self.config.GMAIL_IMAP_SERVER}")
            mail = imaplib.IMAP4_SSL(self.config.GMAIL_IMAP_SERVER)
            
            print(f"[🔍 디버그] 로그인 시도 중: {username}")
            mail.login(username, password)
            
            print(f"[🔍 디버그] 받은편지함 선택 중...")
            mail.select("inbox")
            
            print(f"[✅ IMAP 연결 완료]")
            return mail
        except Exception as e:
            print(f"[❗IMAP 연결 실패] {str(e)}")
            raise
    
    def connect_smtp(self, username, password):
        """SMTP 연결"""
        try:
            server = smtplib.SMTP_SSL(self.config.GMAIL_SMTP_SERVER, self.config.SMTP_PORT)
            server.login(username, password)
            return server
        except Exception as e:
            print(f"[❗SMTP 연결 실패] {str(e)}")
            raise
    
    def fetch_emails(self, username, password, count=5, after_date=None):
        """받은메일 가져오기"""
        self.username = username
        mail = self.connect_imap(username, password)
        
        try:
            mail.select("inbox")  # 받은편지함 선택
            status, data = mail.search(None, "ALL")
            all_mail_ids = data[0].split()
            mail_ids = all_mail_ids[-count:]
            mail_ids.reverse()  # 최신순
            
            emails = []
            for msg_id in mail_ids:
                email_data = self._process_email(mail, msg_id, after_date, mail_type='inbox')
                if email_data:
                    emails.append(email_data)
            
            return emails
        finally:
            mail.close()
            mail.logout()
    
    def fetch_sent_emails(self, username, password, count=5, after_date=None):
        """보낸메일 가져오기 (AI 처리 없음)"""
        self.username = username
        
        try:
            print(f"[🔍 디버그] IMAP 서버 연결 중: {self.config.GMAIL_IMAP_SERVER}")
            mail = imaplib.IMAP4_SSL(self.config.GMAIL_IMAP_SERVER)
            
            print(f"[🔍 디버그] 로그인 시도 중: {username}")
            mail.login(username, password)
            
            # 보낸편지함 선택 (Gmail에서는 '[Gmail]/보낸편지함' 또는 '[Gmail]/Sent Mail')
            folder_selected = False
            
            # 먼저 사용 가능한 폴더 목록 확인
            try:
                print("[🔍 폴더 목록 확인]")
                status, folders = mail.list()
                for folder in folders[:10]:  # 처음 10개만 출력
                    print(f"[📁] {folder.decode()}")
            except Exception as e:
                print(f"[⚠️ 폴더 목록 조회 실패] {str(e)}")
            
            # 폴더명 시도 순서 (실제 Gmail 폴더명 사용)
            folders_to_try = [
                '"[Gmail]/&vPSwuNO4ycDVaA-"',  # Gmail 한글 보낸편지함
                'sent',
                'SENT',
                'Sent', 
                '"[Gmail]/Sent Mail"',
                'INBOX.Sent'
            ]
            
            for folder_name in folders_to_try:
                try:
                    print(f"[🔍 폴더 시도] {folder_name}")
                    result = mail.select(folder_name)
                    if result[0] == 'OK':
                        print(f"[📤 보낸편지함] {folder_name} 선택 완료")
                        folder_selected = True
                        break
                except Exception as e:
                    print(f"[⚠️ 폴더 실패] {folder_name}: {str(e)}")
                    continue
            
            if not folder_selected:
                print("[❗보낸편지함] 모든 폴더 선택 실패")
                return []
            
            status, data = mail.search(None, "ALL")
            all_mail_ids = data[0].split()
            mail_ids = all_mail_ids[-count:]
            mail_ids.reverse()  # 최신순
            
            emails = []
            for msg_id in mail_ids:
                email_data = self._process_email(mail, msg_id, after_date, mail_type='sent')
                if email_data:
                    emails.append(email_data)
            
            print(f"[📤 보낸메일] {len(emails)}개 가져오기 완료")
            return emails
        finally:
            try:
                mail.close()
                mail.logout()
            except Exception as e:
                print(f"[⚠️ IMAP 연결 종료 오류] {str(e)}")
                try:
                    mail.logout()
                except:
                    pass
    
    def _process_email(self, mail, msg_id, after_date=None, mail_type='inbox'):
        """개별 이메일 처리 - Gmail 원본 데이터만 반환"""
        try:
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if not msg_data or not msg_data[0]:
                return None
            
            msg = email_module.message_from_bytes(msg_data[0][1])
            
            # 제목 디코딩
            subject = self._decode_header(msg.get("Subject", ""))
            
            # 발신자 정보
            # From 헤더 디코딩 추가
            raw_from = msg.get("From", "")
            decoded_from = self._decode_header(raw_from)
            name, addr = parseaddr(decoded_from)
            from_field = f"{name} <{addr}>" if name else addr
            
            # 날짜 처리
            raw_date = msg.get("Date", "")
            date_obj, date_str = self._parse_date(raw_date)
            
            # 날짜 필터링
            if after_date and date_obj:
                if date_obj <= after_date:
                    return None
            
            # 본문 추출
            body = self._extract_body(msg)

            # ✅ Message-ID 헤더를 고유 식별자로 사용 (IMAP UID 대신)
            message_id_header = msg.get("Message-ID", "")
            if message_id_header:
                # Message-ID에서 < > 제거하고 해시로 단축
                import hashlib
                clean_id = message_id_header.strip('<>')
                mail_id_str = hashlib.sha256(clean_id.encode()).hexdigest()[:16]  # 16자리 해시
                print(f"[🔍 Message-ID → Hash] {clean_id} → {mail_id_str}")
            else:
                # Message-ID가 없는 경우 fallback (드문 경우)
                imap_uid = str(msg_id.decode()) if isinstance(msg_id, bytes) else str(msg_id)
                mail_id_str = f"imap_{imap_uid}"
                print(f"[⚠️ Message-ID 없음, IMAP UID 사용] {mail_id_str}")
            
            return {
                "id": mail_id_str,  # 문자열로 통일
                "subject": subject,
                "from": from_field,
                "date": date_str,
                "date_obj": date_obj,  # 정확한 날짜 객체 추가
                "body": body,
                "raw_message": msg,
                "mail_type": mail_type  # 'inbox' 또는 'sent'
            }
            
        except Exception as e:
            print(f"[⚠️ 이메일 처리 오류] {str(e)}")
            return None
    
    def _decode_header(self, raw_header):
        """헤더 디코딩"""
        try:
            decoded_parts = decode_header(raw_header)
            if decoded_parts and decoded_parts[0]:
                decoded_header = decoded_parts[0]
                if isinstance(decoded_header[0], bytes):
                    encoding = decoded_header[1] or 'utf-8'
                    return decoded_header[0].decode(encoding)
                else:
                    return str(decoded_header[0])
            return "(제목 없음)"
        except Exception:
            return raw_header if raw_header else "(제목 없음)"
    
    def _parse_date(self, raw_date):
        """날짜 파싱"""
        try:
            date_obj = parsedate_to_datetime(raw_date)
                        
            # 한국시간으로 변환
            if date_obj.tzinfo:
                kst = timezone(timedelta(hours=9))
                date_obj = date_obj.astimezone(kst)  # ← kst로 전부 변환

            date_obj = date_obj.replace(tzinfo=None)                
            date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
            return date_obj, date_str
        except Exception as e:
            print(f"[⚠️ 날짜 파싱 오류] {raw_date}: {e}")
            return None, raw_date[:19] if len(raw_date) >= 19 else raw_date
    
    def _extract_body(self, msg):
        """본문 추출"""
        body = ""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    if (part.get_content_type() == "text/plain" and 
                        not part.get("Content-Disposition")):
                        charset = part.get_content_charset() or "utf-8"
                        body += part.get_payload(decode=True).decode(charset, errors="ignore")
            else:
                charset = msg.get_content_charset() or "utf-8"
                body = msg.get_payload(decode=True).decode(charset, errors="ignore")
            
            return body.strip()
        except Exception:
            return ""
    
    def send_email(self, username, password, to, subject, body, from_header=None, is_html=False):
        """이메일 발송"""
        server = self.connect_smtp(username, password)
        
        try:
            # HTML 또는 일반 텍스트 메시지 생성
            if is_html:
                from email.mime.multipart import MIMEMultipart
                from email.mime.text import MIMEText
                
                msg = MIMEMultipart('alternative')
                msg["Subject"] = subject
                msg["From"] = from_header if from_header else username
                msg["To"] = to
                
                # HTML 파트 추가
                html_part = MIMEText(body, 'html', 'utf-8')
                msg.attach(html_part)
            else:
                msg = MIMEText(body, 'plain', 'utf-8')
                msg["Subject"] = subject
                msg["From"] = from_header if from_header else username
                msg["To"] = to
            
            server.send_message(msg)
            print(f"[📤 메일 전송 성공] {username} -> {to}")
            return True
        finally:
            server.quit()
    
    def search_emails(self, username, password, search_query, max_results=50):
        """이메일 검색"""
        mail = self.connect_imap(username, password)
        
        try:
            status, data = mail.search(None, "ALL")
            all_mail_ids = data[0].split()
            mail_ids = all_mail_ids[-max_results:]
            
            found_emails = []
            
            for msg_id in mail_ids:
                email_data = self._process_email(mail, msg_id)
                if email_data and self._matches_search(email_data, search_query):
                    found_emails.append({
                        "id": email_data["id"],
                        "subject": email_data["subject"][:60] + "..." if len(email_data["subject"]) > 60 else email_data["subject"],
                        "from": email_data["from"][:40] + "..." if len(email_data["from"]) > 40 else email_data["from"],
                        "date": email_data["date"],
                        "preview": email_data["body"][:200] + "..." if len(email_data["body"]) > 200 else email_data["body"]
                    })
                    
                    if len(found_emails) >= 10:
                        break
            
            return found_emails
        finally:
            mail.close()
            mail.logout()
    
    def _matches_search(self, email_data, search_query):
        """검색 쿼리 매칭"""
        search_text = f"{email_data['subject']} {email_data['from']} {email_data['body']}".lower()
        search_lower = search_query.lower()
        
        # 여러 키워드 중 하나라도 매칭되면 포함
        keywords = search_lower.split()
        return any(keyword in search_text for keyword in keywords)
    
    # ✅ AI 요약 기능 제거 - email_routes.py에서 처리

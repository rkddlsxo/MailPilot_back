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
    def __init__(self, config, summarizer=None):
        self.config = config
        self.summarizer = summarizer
    
    def connect_imap(self, username, password):
        """IMAP 연결"""
        try:
            mail = imaplib.IMAP4_SSL(self.config.GMAIL_IMAP_SERVER)
            mail.login(username, password)
            mail.select("inbox")
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
        """이메일 가져오기"""
        self.username = username
        mail = self.connect_imap(username, password)
        
        try:
            status, data = mail.search(None, "ALL")
            all_mail_ids = data[0].split()
            mail_ids = all_mail_ids[-count:]
            mail_ids.reverse()  # 최신순
            
            emails = []
            for msg_id in mail_ids:
                email_data = self._process_email(mail, msg_id, after_date)
                if email_data:
                    emails.append(email_data)
            
            return emails
        finally:
            mail.close()
            mail.logout()
    
    def _process_email(self, mail, msg_id, after_date=None):
        """개별 이메일 처리"""
        try:
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if not msg_data or not msg_data[0]:
                return None
            
            msg = email_module.message_from_bytes(msg_data[0][1])
            
            # 제목 디코딩
            subject = self._decode_header(msg.get("Subject", ""))
            
            # 발신자 정보
            name, addr = parseaddr(msg.get("From"))
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

            # 요약 생성
            summary_text = ""
            try:
                if self.summarizer and body:
                    safe_text = body[:1000]
                    if len(safe_text) < 50:
                        summary_text = safe_text
                    else:
                        summary_text = self.summarizer(
                            safe_text,
                            max_length=80,
                            min_length=30,
                            do_sample=False
                        )[0]["summary_text"]
                else:
                    summary_text = "(요약 없음)"
            except Exception as e:
                print(f"[⚠️ 요약 실패] {str(e)}")
                summary_text = "(요약 실패)"

            # 디비 중복체크
            mail_id_str = str(msg_id.decode()) if isinstance(msg_id, bytes) else str(msg_id)
            existing = Mail.query.filter_by(user_email=self.username, mail_id=mail_id_str).first()
            
            # 디비 저장
            if not existing: 
                new_mail = Mail(
                    user_email=self.username,
                    mail_id=mail_id_str,
                    subject=subject,
                    from_=from_field,
                    body=body,
                    raw_message=msg.as_string(),
                    date=date_obj,
                    summary=summary_text
                )
                db.session.add(new_mail)
                db.session.commit()
                print(f"[📥 저장 완료] {self.username} → {subject[:30]}...")


            return {
                "id": int(msg_id.decode()) if isinstance(msg_id, bytes) else int(msg_id),
                "subject": subject,
                "from": from_field,
                "date": date_str,
                "body": body,
                "raw_message": msg
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
        except:
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
    
    def send_email(self, username, password, to, subject, body):
        """이메일 발송"""
        server = self.connect_smtp(username, password)
        
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = username
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

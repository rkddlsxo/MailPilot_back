from flask import Flask, request, jsonify
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from flask_cors import CORS
import imaplib
import email
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# 요약 모델 로딩
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")

@app.route('/api/summary', methods=['POST'])
def summary():
    try:
        data = request.get_json()
        username = data.get("email")
        app_password = data.get("app_password")

        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, app_password)
        mail.select("inbox")

        N = 5
        status, data = mail.search(None, "ALL")
        mail_ids = data[0].split()[-N:]

        emails = []

        for index, msg_id in enumerate(mail_ids):
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            raw_msg = msg_data[0][1]
            msg = email.message_from_bytes(raw_msg)

            # 제목 디코딩
            raw_subject = msg.get("Subject", "")
            decoded_subject = decode_header(raw_subject)[0]
            subject = decoded_subject[0].decode(decoded_subject[1]) if isinstance(decoded_subject[0], bytes) else decoded_subject[0]

            # 보내는 사람
            name, addr = parseaddr(msg.get("From"))
            from_field = f"{name} <{addr}>" if name else addr

            # 날짜
            raw_date = msg.get("Date", "")[:25]
            try:
                date_obj = parsedate_to_datetime(raw_date)
                date_str = date_obj.strftime("%Y-%m-%d")
            except:
                date_str = raw_date[:10]

            # 본문
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                        charset = part.get_content_charset() or "utf-8"
                        body += part.get_payload(decode=True).decode(charset, errors="ignore")
            else:
                charset = msg.get_content_charset() or "utf-8"
                body = msg.get_payload(decode=True).decode(charset, errors="ignore")

            body = body.strip()
            if not body:
                continue

            # 요약 실행 (원하는 경우 summary 필드로 따로 추가 가능)
            summary_text = summarizer(body[:3000], max_length=80, min_length=30, do_sample=False)[0]["summary_text"]

            # 태그 추정 (IMAP에서는 정확한 라벨 정보를 받기 어려움 → 간이 처리)
            typ, flag_data = mail.fetch(msg_id, "(FLAGS)")
            flags_bytes = flag_data[0]
            flags_str = flags_bytes.decode() if isinstance(flags_bytes, bytes) else str(flags_bytes)

            tag = "받은"
            if "\\Important" in flags_str:
                tag = "중요"
            elif "\\Junk" in flags_str or "\\Spam" in flags_str:
                tag = "스팸"

            emails.append({
                "id": index + 1,
                "subject": subject,
                "from": from_field,
                "date": date_str,
                "body": body[:1000],  # 너무 길면 자름
                "tag": tag,
                "summary": summary_text 
            })

        return jsonify({"emails": emails})

    except Exception as e:
        print("[❗에러 발생]", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/test', methods=['POST'])
def test():
    data = request.get_json()
    text = data.get("text", "")
    return jsonify({"message": f"✅ 백엔드 정상 작동: {text[:20]}..."})


@app.route('/', methods=['GET'])
def health_check():
    return "✅ 백엔드 정상 작동 중", 200


if __name__ == '__main__':
    app.run(debug=True)




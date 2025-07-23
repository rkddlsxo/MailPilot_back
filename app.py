from flask import Flask, request, jsonify
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from flask_cors import CORS
from datetime import datetime
import imaplib
import email
import smtplib
import traceback
from email.mime.text import MIMEText
from transformers import pipeline
from nomic import embed
from sklearn.metrics.pairwise import cosine_similarity
from nomic import login

login(token="nk-QV0H1frBySMJ8TH8Vz4_smZsg_iurT-G0EH_HMnrMKg")

candidate_labels = [
    "university.",
    "spam mail.",
    "company.",
    "security alert."
]

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

        # 문자열 날짜를 datetime 객체로 변환
        after_date = data.get("after")

        after_dt = None
        if after_date:
            try:
                after_date_clean = after_date.replace("Z", "+00:00")
                after_dt = datetime.fromisoformat(after_date_clean)
                after_dt = after_dt.replace(tzinfo=None)
            except Exception as e:
                print("[⚠️ after_date 파싱 실패]", e)

        # 메일 서버 연결
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, app_password)
        mail.select("inbox")

        N = 5  # 최근 5개 메일만
        status, data = mail.search(None, "ALL")
        mail_ids = data[0].split()[-N:]

        emails = []

        for index, msg_id in enumerate(mail_ids):
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if not msg_data or not msg_data[0]:
                continue

            raw_msg = msg_data[0][1]
            msg = email.message_from_bytes(raw_msg)

            # 제목 디코딩
            raw_subject = msg.get("Subject", "")
            decoded_parts = decode_header(raw_subject)
            if decoded_parts:
                decoded_subject = decoded_parts[0]
                subject = decoded_subject[0].decode(decoded_subject[1]) if isinstance(decoded_subject[0], bytes) else decoded_subject[0]
            else:
                subject = "(제목 없음)"

            # 보내는 사람
            name, addr = parseaddr(msg.get("From"))
            from_field = f"{name} <{addr}>" if name else addr

            # 날짜
            raw_date = msg.get("Date", "")[:25]
            try:
                date_obj = parsedate_to_datetime(raw_date)
                date_obj = date_obj.replace(tzinfo=None)
                date_str = date_obj.strftime("%Y-%m-%d")
            except:
                date_obj = None
                date_str = raw_date[:10]

            # after_date 이후 메일만 필터링
            if after_dt and date_obj:
                if date_obj <= after_dt:
                    continue

            # 본문 추출
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
                body = ""  # 안전하게 빈 문자열로 처리

            # 분류 실행
            try:
                text_inputs = [body] + candidate_labels
                result = embed.text(text_inputs, model='nomic-embed-text-v1', task_type='classification')
                embedding_list = result['embeddings']
                email_embedding = [embedding_list[0]]
                label_embeddings = embedding_list[1:]
                scores = cosine_similarity(email_embedding, label_embeddings)[0]
                best_index = scores.argmax()
                classification_tag = candidate_labels[best_index]
                confidence = scores[best_index]
                print("[💩 분류 성공]", classification_tag)
            except Exception as e:
                print("[⚠️ 분류 실패]", str(e))
                classification_tag = "unknown"

            # 요약 실행 (안전 처리)
            try:
                if not body:
                    summary_text = "(본문 없음)"
                else:
                    safe_text = body[:1000]  # 모델 입력 제한 준수
                    if len(safe_text) < 50:
                        summary_text = safe_text
                    else:
                        summary_text = summarizer(
                            safe_text,
                            max_length=80,
                            min_length=30,
                            do_sample=False
                        )[0]["summary_text"]
            except Exception as e:
                print("[⚠️ 요약 실패]", str(e))
                summary_text = body[:150] + "..." if body else "(요약 실패)"

            # 태그 추정
            typ, flag_data = mail.fetch(msg_id, "(FLAGS)")
            if flag_data and flag_data[0]:
                flags_bytes = flag_data[0]
                flags_str = flags_bytes.decode() if isinstance(flags_bytes, bytes) else str(flags_bytes)
            else:
                flags_str = ""

            tag = "받은"
            if "\\Important" in flags_str:
                tag = "중요"
            elif "\\Junk" in flags_str or "\\Spam" in flags_str:
                tag = "스팸"

            # 메일 객체 추가
            emails.append({
                "id": index + 1,
                "subject": subject,
                "from": from_field,
                "date": date_str,
                "body": body[:1000],
                "tag": tag,
                "summary": summary_text,
                "classification": classification_tag,
            })

        return jsonify({"emails": emails})

    except Exception as e:
        print("[❗에러 발생]", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/test', methods=['POST'])
def test():
    data = request.get_json()
    text = data.get("text", "")
    return jsonify({"message": f"✅ 백엔드 정상 작동: {text[:20]}..."})

@app.route("/api/send", methods=["POST"])
def send_email():
    try:
        data = request.get_json()
        print("✅ 받은 데이터:", data)  # POST 요청이 실제로 왔는지 확인

        sender_email = data["email"]
        app_password = data["app_password"]
        to = data["to"]
        subject = data["subject"]
        body = data["body"]

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()

        return jsonify({"message": "✅ 메일 전송 성공"}), 200

    except Exception as e:
        print("[❗메일 전송 실패]", str(e))  # 상세 에러 출력
        return jsonify({"error": str(e)}), 500
 


@app.route('/', methods=['GET'])
def health_check():
    return "✅ 백엔드 정상 작동 중", 200





if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)




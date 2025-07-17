from flask import Flask, request, jsonify
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

        # Gmail IMAP 접속
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, app_password)
        mail.select("inbox")

        # 최신 메일 N개 가져오기
        N = 3
        status, data = mail.search(None, "ALL")
        mail_ids = data[0].split()[-N:]

        summaries = []

        for msg_id in mail_ids:
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])

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

            if not body.strip():
                continue

            # 요약 실행 (입력 길이 제한 포함)
            result = summarizer(body[:3000], max_length=80, min_length=30, do_sample=False)
            summaries.append(result[0]["summary_text"])

        return jsonify({"summaries": summaries})

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




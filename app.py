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
import os
from huggingface_hub import InferenceClient

login(token="nk-QV0H1frBySMJ8TH8Vz4_smZsg_iurT-G0EH_HMnrMKg")

# Hugging Face 토큰 설정
os.environ['HF_TOKEN'] = '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'

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

def build_ai_reply_prompt(sender, subject, body):
    """AI 답장을 위한 프롬프트 생성"""
    return f"""
You are a helpful email assistant that writes professional email replies.

Please read the following email and write a polite, professional reply in English:

---
From: {sender}
Subject: {subject}
Body: {body}
---

Instructions:
1. Identify the purpose of the email (invitation, question, information request, scheduling, etc.)
2. Write a concise (3-4 sentences), polite reply that directly addresses the purpose
3. Use a friendly yet professional tone
4. Only output the reply text (no analysis, no quotes, no original email content)

Reply:
""".strip()

@app.route('/api/generate-ai-reply', methods=['POST'])
def generate_ai_reply():
    """AI 답장 생성 API"""
    try:
        data = request.get_json()
        sender = data.get('sender', '')
        subject = data.get('subject', '')
        body = data.get('body', '')
        
        print(f"[🤖 AI 답장 요청] From: {sender}, Subject: {subject[:50]}...")
        
        if not all([sender, subject, body]):
            return jsonify({'error': '발신자, 제목, 본문이 모두 필요합니다.'}), 400
        
        # Hugging Face 토큰 확인
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            return jsonify({'error': 'HF_TOKEN 환경 변수가 설정되어 있지 않습니다.'}), 500
        
        # InferenceClient 생성
        client = InferenceClient(
            model="Qwen/Qwen2.5-7B-Instruct",
            token=hf_token
        )
        
        # 프롬프트 생성
        user_prompt = build_ai_reply_prompt(sender, subject, body)
        
        # AI 답장 생성
        messages = [
            {"role": "system", "content": "You are a helpful email assistant that writes professional email replies."},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.7
        )
        
        # 답장 텍스트 추출
        ai_reply = response.choices[0].message.content.strip()
        
        print(f"[✅ AI 답장 생성 완료] 길이: {len(ai_reply)}자")
        
        return jsonify({
            'success': True,
            'ai_reply': ai_reply
        })
        
    except Exception as e:
        print(f"[❗AI 답장 생성 실패] {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'AI 답장 생성 실패: {str(e)}'}), 500

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
                print(f"[📅 필터링 기준] {after_dt} 이후 메일만 가져옴")
            except Exception as e:
                print("[⚠️ after_date 파싱 실패]", e)

        # 메일 서버 연결
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, app_password)
        mail.select("inbox")

        # ✅ 메일 수 동적 결정
        if after_dt:
            # 새로고침인 경우: 최근 10개 메일 검색
            N = 10
            print(f"[🔄 새로고침] 최근 {N}개 메일에서 {after_dt} 이후 메일 검색")
        else:
            # 첫 로딩인 경우: 최근 5개 메일
            N = 5
            print(f"[🆕 첫 로딩] 최근 {N}개 메일 가져옴")

        status, data = mail.search(None, "ALL")
        all_mail_ids = data[0].split()
        
        # ✅ 최신 메일부터 처리하도록 순서 수정
        mail_ids = all_mail_ids[-N:]  # 마지막 N개
        mail_ids.reverse()  # 최신 메일이 먼저 오도록 뒤집기

        emails = []
        processed_count = 0

        for msg_id in mail_ids:
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

            # 날짜 처리 개선
            raw_date = msg.get("Date", "")
            try:
                date_obj = parsedate_to_datetime(raw_date)
                date_obj = date_obj.replace(tzinfo=None)
                date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")  # 시간 정보도 포함
            except:
                date_obj = None
                date_str = raw_date[:19] if len(raw_date) >= 19 else raw_date

            # ✅ after_date 필터링 로직 개선
            if after_dt and date_obj:
                if date_obj <= after_dt:
                    print(f"[⏭️ 건너뛰기] {date_str} (기준: {after_dt})")
                    continue
                else:
                    print(f"[✅ 포함] {date_str} - {subject[:30]}...")

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
                body = ""

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
                print(f"[🏷️ 분류] {classification_tag} (신뢰도: {confidence:.3f})")
            except Exception as e:
                print("[⚠️ 분류 실패]", str(e))
                classification_tag = "unknown"

            # 요약 실행
            try:
                if not body:
                    summary_text = "(본문 없음)"
                else:
                    safe_text = body[:1000]
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

            # ✅ 메일 객체 추가 (ID를 msg_id 기반으로 생성)
            emails.append({
                "id": int(msg_id.decode()) if isinstance(msg_id, bytes) else int(msg_id),
                "subject": subject,
                "from": from_field,
                "date": date_str,
                "body": body[:1000],
                "tag": tag,
                "summary": summary_text,
                "classification": classification_tag,
            })
            
            processed_count += 1

        # ✅ 백엔드에서도 날짜순 정렬 (최신 먼저)
        emails.sort(key=lambda x: x['date'], reverse=True)
        
        print(f"[📊 결과] 총 {processed_count}개 메일 처리 완료")
        if emails:
            print(f"[📅 범위] {emails[-1]['date']} ~ {emails[0]['date']}")

        return jsonify({"emails": emails})

    except Exception as e:
        print("[❗에러 발생]", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        email = data.get("email", "")
        app_password = data.get("app_password", "")
        
        print(f"[🤖 챗봇 요청] {user_input}")
        
        if not user_input:
            return jsonify({"error": "입력이 비어있습니다."}), 400
        
        # Define candidate labels (기능 라벨)
        candidate_labels = [
            "correct the vocabulary, spelling",
            "image generation using text", 
            "find something",
            "email search for a person"
        ]
        
        # Generate embeddings
        text_inputs = [user_input] + candidate_labels
        result = embed.text(text_inputs, model='nomic-embed-text-v1', task_type='classification')
        
        # Compare embeddings
        embedding_list = result['embeddings']
        email_embedding = [embedding_list[0]]
        label_embeddings = embedding_list[1:]
        
        # Cosine Similarity
        scores = cosine_similarity(email_embedding, label_embeddings)[0]
        best_index = scores.argmax()
        best_score = scores[best_index]
        best_label = candidate_labels[best_index]
        
        print(f"[🎯 분류 결과] {best_label} (유사도: {best_score:.4f})")
        
        # Threshold decision
        threshold = 0.3
        
        if best_score >= threshold:
            if best_label == "correct the vocabulary, spelling":
                response = "📝 문법 및 맞춤법 교정 기능입니다.\n\n교정하고 싶은 텍스트를 입력해주세요. 현재는 기본 응답을 제공하고 있으며, 향후 전문 교정 모델이 추가될 예정입니다."
                action = "grammar_correction"
                
            elif best_label == "image generation using text":
                response = "🎨 텍스트를 이미지로 변환하는 기능입니다.\n\n생성하고 싶은 이미지에 대한 설명을 입력해주세요. 현재는 기본 응답을 제공하고 있으며, 향후 이미지 생성 모델이 추가될 예정입니다."
                action = "image_generation"
                
            elif best_label == "find something":
                response = "🔍 특정 키워드가 포함된 메일을 검색하는 기능입니다.\n\n찾고 싶은 키워드나 내용을 알려주세요. 메일 제목, 발신자, 내용에서 검색해드립니다."
                action = "email_search"
                
            elif best_label == "email search for a person":
                response = "👤 특정 사람의 메일을 검색하는 기능입니다.\n\n찾고 싶은 사람의 이름이나 이메일 주소를 알려주세요."
                action = "person_search"
        else:
            response = "❓ 요청을 정확히 이해하지 못했습니다.\n\n다음 기능들을 이용해보세요:\n• 문법/맞춤법 교정\n• 이미지 생성\n• 메일 검색\n• 특정 사람 메일 찾기"
            action = "unknown"
        
        return jsonify({
            "response": response,
            "action": action,
            "confidence": float(best_score),
            "detected_intent": best_label
        }), 200
        
    except Exception as e:
        print("[❗챗봇 오류]", str(e))
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
        print("✅ 받은 데이터:", data)

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
        print("[❗메일 전송 실패]", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return "✅ 백엔드 정상 작동 중", 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
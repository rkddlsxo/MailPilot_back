from flask import Flask, request, jsonify, session
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from flask_cors import CORS
from datetime import datetime
import imaplib
import smtplib
import traceback
from email.mime.text import MIMEText
from transformers import pipeline
from nomic import embed
from sklearn.metrics.pairwise import cosine_similarity
from nomic import login
import os
import hashlib
import uuid
from huggingface_hub import InferenceClient
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pprint import pprint
import re
import email as email_module


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
CORS(app, supports_credentials=True)  # 세션 쿠키 지원
app.secret_key = 'your-secret-key-here'  # 세션 암호화용 키

# ✅ Qwen 모델 전역 변수 (한 번만 로딩하여 성능 최적화)
qwen_model = None
qwen_tokenizer = None

# 사용자별 데이터 저장소 (실제 운영에서는 Redis나 데이터베이스 사용 권장)
user_sessions = {}

# ===== 3. 여기에 Qwen 관련 함수들 추가 =====
def load_qwen_model():
    """Qwen 모델을 로딩하는 함수"""
    global qwen_model, qwen_tokenizer
    
    if qwen_model is None:
        print("[🤖 Qwen 모델 로딩 시작]")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_id = "Qwen/Qwen1.5-1.8B-Chat"
            
            qwen_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            qwen_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            qwen_model.eval()
            print("[✅ Qwen 모델 로딩 완료]")
        except Exception as e:
            print(f"[❗Qwen 모델 로딩 실패] {str(e)}")
            # Qwen 로딩 실패해도 다른 기능은 정상 작동하도록

def extract_search_target_with_qwen(text):
    """Qwen을 이용하여 검색 대상 추출"""
    global qwen_model, qwen_tokenizer
    
    # 모델이 로딩되지 않았다면 로딩 시도
    if qwen_model is None:
        load_qwen_model()
    
    # 모델 로딩에 실패한 경우 간단한 키워드 추출로 fallback
    if qwen_model is None:
        print("[⚠️ Qwen 모델 없음 - 간단 추출 사용]")
        words = text.split()
        return " ".join(words[-2:]) if len(words) >= 2 else text
    
    try:
        prompt = (
            "<|im_start|>system\nYou are an email assistant. "
            "Your job is to extract the email address or name the user is referring to. "
            "You must always respond in the format: The user is referring to ... \n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)
        
        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=qwen_tokenizer.eos_token_id
            )
        
        decoded_output = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # "assistant" 이후 텍스트만 가져옴
        if "assistant" in decoded_output:
            after_assistant = decoded_output.split("assistant")[-1].strip()
            
            # "The user is referring to" 뒷부분만 추출
            prefix = "The user is referring to "
            if prefix in after_assistant:
                result = after_assistant.split(prefix)[-1].strip().rstrip(".")
                return result
        
        return decoded_output.strip()
    except Exception as e:
        print(f"[⚠️ Qwen 추출 오류] {str(e)}")
        # 오류 시 간단한 키워드 추출로 fallback
        words = text.split()
        return " ".join(words[-2:]) if len(words) >= 2 else text

def search_emails_by_target(emails, search_target):
    """이메일 리스트에서 검색 대상으로 필터링"""
    results = []
    search_lower = search_target.lower()
    
    for mail in emails:
        # from 필드에서 검색
        if search_lower in mail["from"].lower():
            results.append(mail)
        # 제목에서도 검색
        elif search_lower in mail["subject"].lower():
            results.append(mail)
        # 이메일 주소만 추출해서 검색
        elif "@" in search_target:
            # 이메일 주소 패턴 매칭
            email_pattern = r'<([^>]+)>'
            email_match = re.search(email_pattern, mail["from"])
            if email_match and search_lower in email_match.group(1).lower():
                results.append(mail)
    
    return results

def get_session_id():
    """세션 ID 생성 또는 가져오기"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_user_key(email):
    """이메일 기반 사용자 키 생성"""
    return hashlib.md5(email.encode()).hexdigest()

def clear_user_session(email):
    """특정 사용자의 세션 데이터 삭제"""
    user_key = get_user_key(email)
    if user_key in user_sessions:
        del user_sessions[user_key]
        print(f"[🗑️ 세션 삭제] {email} 사용자 데이터 삭제됨")

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

# ===== 4. 여기에 새로운 API 엔드포인트 추가 =====
# 기존 @app.route('/api/email-search', methods=['POST']) 함수를 이것으로 교체하세요

# /api/email-search 함수를 이것으로 교체하세요

@app.route('/api/email-search', methods=['POST'])
def email_search():
    """이메일 검색 API - 변수명 충돌 해결"""
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        user_email = data.get("email", "")  # ✅ 변수명 변경
        app_password = data.get("app_password", "")
        
        print(f"[🔍 이메일 검색 요청] 사용자: {user_email}, 입력: {user_input}")
        
        if not all([user_input, user_email, app_password]):
            return jsonify({"error": "사용자 입력, 이메일, 앱 비밀번호가 모두 필요합니다."}), 400
        
        # 사용자 세션 확인
        user_key = get_user_key(user_email)
        if user_key not in user_sessions:
            return jsonify({"error": "로그인이 필요합니다."}), 401
        
        print("[🎯 실제 메일 검색 시작]")
        
        # Qwen을 이용해 검색 대상 추출
        try:
            search_target = extract_search_target_with_qwen(user_input)
            print(f"[🎯 검색 대상 추출] '{search_target}'")
        except Exception as e:
            print(f"[⚠️ Qwen 추출 실패] {str(e)}")
            # 간단한 이메일 추출 fallback
            import re
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails_found = re.findall(email_pattern, user_input)
            if emails_found:
                search_target = emails_found[0]
            else:
                words = user_input.split()
                search_target = " ".join(words[-2:]) if len(words) >= 2 else user_input
        
        print(f"[🔍 최종 검색 대상] '{search_target}'")
        
        # 메일 서버 연결 및 검색
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(user_email, app_password)  # ✅ 변수명 수정
            mail.select("inbox")
            print("[✅ 메일 서버 연결 성공]")
            
            # 검색 범위 설정
            N = 100
            status, data_result = mail.search(None, "ALL")
            all_mail_ids = data_result[0].split()
            mail_ids = all_mail_ids[-N:]  # 최근 N개
            
            print(f"[📊 검색 범위] 총 {len(all_mail_ids)}개 중 최근 {len(mail_ids)}개 검색")
            
            emails_found = []
            processed_count = 0
            
            for msg_id in mail_ids:
                try:
                    _, msg_data = mail.fetch(msg_id, "(RFC822)")
                    if not msg_data or not msg_data[0]:
                        continue
                        
                    # ✅ 올바른 모듈 사용
                    msg = email_module.message_from_bytes(msg_data[0][1])
                    processed_count += 1
                    
                    # 제목 디코딩
                    raw_subject = msg.get("Subject", "")
                    try:
                        decoded_parts = decode_header(raw_subject)
                        if decoded_parts and decoded_parts[0]:
                            decoded_subject = decoded_parts[0]
                            subject_bytes = decoded_subject[0]
                            subject_encoding = decoded_subject[1]
                            
                            if isinstance(subject_bytes, bytes):
                                if subject_encoding is None:
                                    subject_encoding = 'utf-8'
                                try:
                                    subject = subject_bytes.decode(subject_encoding)
                                except (UnicodeDecodeError, LookupError):
                                    for fallback_encoding in ['utf-8', 'latin-1', 'cp949', 'euc-kr']:
                                        try:
                                            subject = subject_bytes.decode(fallback_encoding)
                                            break
                                        except (UnicodeDecodeError, LookupError):
                                            continue
                                    else:
                                        subject = subject_bytes.decode('utf-8', errors='ignore')
                            else:
                                subject = str(subject_bytes)
                        else:
                            subject = "(제목 없음)"
                    except Exception as e:
                        subject = raw_subject if raw_subject else "(제목 없음)"
                    
                    # 발신자 정보
                    name, addr = parseaddr(msg.get("From"))
                    from_field = f"{name} <{addr}>" if name else addr
                    
                    # 날짜 처리
                    raw_date = msg.get("Date", "")
                    try:
                        date_obj = parsedate_to_datetime(raw_date)
                        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        date_str = raw_date[:19] if len(raw_date) >= 19 else raw_date
                    
                    # 본문 추출
                    body = ""
                    try:
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                                    charset = part.get_content_charset() or "utf-8"
                                    body += part.get_payload(decode=True).decode(charset, errors="ignore")
                        else:
                            charset = msg.get_content_charset() or "utf-8"
                            body = msg.get_payload(decode=True).decode(charset, errors="ignore")
                        
                        body = body.strip()
                    except Exception as e:
                        body = ""
                    
                    # 검색 대상과 매칭 확인
                    search_in = f"{subject} {from_field} {body}".lower()
                    search_lower = search_target.lower()
                    
                    # 이메일 주소나 이름으로 검색
                    if (search_lower in search_in or 
                        any(part.strip() in search_in for part in search_lower.split() if part.strip())):
                        
                        emails_found.append({
                            "id": int(msg_id.decode()) if isinstance(msg_id, bytes) else int(msg_id),
                            "subject": subject,
                            "from": from_field,
                            "date": date_str,
                            "body": body[:500]  # 처음 500자만
                        })
                        
                        print(f"[✅ 매칭 발견] {from_field} -> {subject[:30]}...")
                        
                        if len(emails_found) >= 10:  # 최대 10개
                            break
                            
                except Exception as e:
                    print(f"[⚠️ 메일 처리 오류] {str(e)}")
                    continue
            
            mail.close()
            mail.logout()
            
            print(f"[📊 검색 완료] {processed_count}개 처리, {len(emails_found)}개 발견")
            
            return jsonify({
                "success": True,
                "search_target": search_target,
                "results": emails_found,
                "total_searched": processed_count,
                "found_count": len(emails_found),
                "confidence": 1.0,
                "detected_intent": "email_search_completed"
            })
            
        except Exception as e:
            print(f"[❗메일 서버 오류] {str(e)}")
            return jsonify({
                "success": False,
                "error": f"메일 서버 연결 오류: {str(e)}",
                "search_target": search_target if 'search_target' in locals() else user_input
            }), 500
            
    except Exception as e:
        print(f"[❗이메일 검색 오류] {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/login', methods=['POST'])
def login_user():
    """사용자 로그인 - 이전 세션 데이터 삭제"""
    try:
        data = request.get_json()
        email = data.get('email', '')
        
        if email:
            # 이전 사용자 데이터 삭제
            clear_user_session(email)
            
            # 새 세션 생성
            session_id = get_session_id()
            user_key = get_user_key(email)
            
            # 사용자별 세션 초기화
            user_sessions[user_key] = {
                'email': email,
                'session_id': session_id,
                'last_emails': [],
                'login_time': datetime.now().isoformat()
            }
            
            print(f"[🔑 로그인] {email} - 새 세션 생성: {session_id[:8]}...")
            
            return jsonify({
                'success': True,
                'message': '로그인 성공',
                'session_id': session_id
            })
        else:
            return jsonify({'error': '이메일이 필요합니다.'}), 400
            
    except Exception as e:
        print(f"[❗로그인 실패] {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout_user():
    """사용자 로그아웃 - 세션 데이터 삭제"""
    try:
        data = request.get_json()
        email = data.get('email', '')
        
        if email:
            clear_user_session(email)
            session.clear()  # Flask 세션도 삭제
            
            return jsonify({
                'success': True,
                'message': '로그아웃 성공'
            })
        else:
            return jsonify({'error': '이메일이 필요합니다.'}), 400
            
    except Exception as e:
        print(f"[❗로그아웃 실패] {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-ai-reply', methods=['POST'])
def generate_ai_reply():
    """AI 답장 생성 API"""
    try:
        data = request.get_json()
        sender = data.get('sender', '')
        subject = data.get('subject', '')
        body = data.get('body', '')
        current_user_email = data.get('email', '')  # 현재 사용자 이메일 추가
        
        print(f"[🤖 AI 답장 요청] User: {current_user_email}, From: {sender}, Subject: {subject[:50]}...")
        
        if not all([sender, subject, body, current_user_email]):
            return jsonify({'error': '발신자, 제목, 본문, 사용자 이메일이 모두 필요합니다.'}), 400
        
        # 사용자 세션 확인
        user_key = get_user_key(current_user_email)
        if user_key not in user_sessions:
            return jsonify({'error': '로그인이 필요합니다.'}), 401
        
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
        
        print(f"[✅ AI 답장 생성 완료] User: {current_user_email}, 길이: {len(ai_reply)}자")
        
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

        # 사용자 키 생성 및 세션 확인
        user_key = get_user_key(username)
        
        print(f"[📧 메일 요청] 사용자: {username}")
        
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

        # 메일 수 동적 결정
        if after_dt:
            N = 10
            print(f"[🔄 새로고침] 최근 {N}개 메일에서 {after_dt} 이후 메일 검색")
        else:
            N = 5
            print(f"[🆕 첫 로딩] 최근 {N}개 메일 가져옴")

        status, data = mail.search(None, "ALL")
        all_mail_ids = data[0].split()
        
        # 최신 메일부터 처리하도록 순서 수정
        mail_ids = all_mail_ids[-N:]
        mail_ids.reverse()

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

            # 날짜 처리
            raw_date = msg.get("Date", "")
            try:
                date_obj = parsedate_to_datetime(raw_date)
                date_obj = date_obj.replace(tzinfo=None)
                date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
            except:
                date_obj = None
                date_str = raw_date[:19] if len(raw_date) >= 19 else raw_date

            # after_date 필터링
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

            # 메일 객체 추가
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

        # 백엔드에서도 날짜순 정렬 (최신 먼저)
        emails.sort(key=lambda x: x['date'], reverse=True)
        
        # 사용자별 세션에 메일 데이터 저장
        if user_key not in user_sessions:
            user_sessions[user_key] = {}
        
        user_sessions[user_key]['last_emails'] = emails
        user_sessions[user_key]['last_update'] = datetime.now().isoformat()
        
        print(f"[📊 결과] 사용자: {username}, 총 {processed_count}개 메일 처리 완료")
        if emails:
            print(f"[📅 범위] {emails[-1]['date']} ~ {emails[0]['date']}")

        return jsonify({
            "emails": emails,
            "user_session": user_key[:8] + "...",  # 디버그용
            "cache_info": f"세션에 {len(emails)}개 메일 저장됨"
        })

    except Exception as e:
        print("[❗에러 발생]", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()
        user_email = data.get("email", "")
        app_password = data.get("app_password", "")
        
        print(f"[🤖 챗봇 요청] 사용자: {email}, 입력: {user_input}")
        
        if not user_input:
            return jsonify({"error": "입력이 비어있습니다."}), 400
        
        # 사용자 세션 확인
        user_key = get_user_key(user_email)
        if user_key not in user_sessions:
            print(f"[⚠️ 세션 없음] {email} 사용자의 세션이 없습니다.")
            return jsonify({"error": "로그인이 필요합니다."}), 401
        
        # ✅ 1. 의도 분류 (Python 코드와 동일)
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
        email_embedding = [embedding_list[0]]  # 첫 번째: 사용자 입력
        label_embeddings = embedding_list[1:]  # 나머지: 기능 라벨
        
        # Cosine Similarity
        scores = cosine_similarity(email_embedding, label_embeddings)[0]
        best_index = scores.argmax()
        best_score = scores[best_index]
        best_label = candidate_labels[best_index]
        
        print(f"[🎯 분류 결과] 사용자: {email}, 의도: {best_label} (유사도: {best_score:.4f})")
        
        # ✅ 2. Threshold decision
        threshold = 0.3
        
        if best_score >= threshold:
            # ✅ 3. 각 기능별 실제 구현
            if best_label == "correct the vocabulary, spelling":
                response = handle_grammar_correction(user_input)
                action = "grammar_correction"
                
            elif best_label == "image generation using text":
                response = handle_image_generation(user_input)
                action = "image_generation"
                
            elif best_label == "find something":
                response = handle_general_search(user_input, user_email, app_password)
                action = "email_search"
                
            elif best_label == "email search for a person":
                response = handle_person_search(user_input, user_email, app_password)
                action = "person_search"
        else:
            response = "❓ 요청을 이해하지 못했습니다. 다른 표현을 시도해주세요.\n\n다음 기능들을 이용해보세요:\n• 문법/맞춤법 교정\n• 이미지 생성\n• 메일 검색\n• 특정 사람 메일 찾기"
            action = "unknown"
        
        return jsonify({
            "response": response,
            "action": action,
            "confidence": float(best_score),
            "detected_intent": best_label,
            "user_session": user_key[:8] + "..."
        }), 200
        
    except Exception as e:
        print("[❗챗봇 오류]", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    # ✅ 4. 각 기능별 핸들러 함수들

    def handle_grammar_correction(user_input):
        """문법 및 맞춤법 교정 기능"""
    try:
        # 교정할 텍스트 추출 (간단한 방법)
        correction_text = user_input
        
        # "교정해주세요", "맞춤법" 등의 단어 제거
        remove_words = ["교정해주세요", "교정해줘", "맞춤법", "문법", "correct", "spelling"]
        for word in remove_words:
            correction_text = correction_text.replace(word, "").strip()
        
        if not correction_text:
            return "📝 **문법 및 맞춤법 교정**\n\n교정하고 싶은 텍스트를 입력해주세요.\n\n예시: '안녕하세요. 제가 오늘 회의에 참석못할것 같습니다' 교정해주세요"
        
        # Hugging Face 모델을 이용한 교정 (간단한 예시)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                client = InferenceClient(token=hf_token)
                
                prompt = f"""다음 텍스트의 맞춤법과 문법을 교정해주세요:

원본: {correction_text}

교정된 텍스트:"""
                
                response = client.text_generation(
                    prompt,
                    model="microsoft/DialoGPT-medium",
                    max_new_tokens=100
                )
                
                corrected_text = response.strip()
                
                return f"📝 **문법 및 맞춤법 교정**\n\n**원본:**\n{correction_text}\n\n**교정된 텍스트:**\n{corrected_text}\n\n💡 AI가 제안한 교정안입니다. 검토 후 사용해주세요."
                
            except Exception as e:
                print(f"[⚠️ 교정 모델 오류] {str(e)}")
        
        # 기본 응답
        return f"📝 **문법 및 맞춤법 교정**\n\n입력된 텍스트: '{correction_text}'\n\n⚠️ 현재는 기본 응답을 제공하고 있습니다.\n향후 전문 교정 모델이 추가될 예정입니다."
        
    except Exception as e:
        print(f"[❗문법 교정 오류] {str(e)}")
        return "❌ 문법 교정 처리 중 오류가 발생했습니다."


def handle_image_generation(user_input):
    """이미지 생성 기능"""
    try:
        # 이미지 생성 프롬프트 추출
        image_prompt = user_input
        
        # "이미지 생성", "그려줘" 등의 단어 제거
        remove_words = ["이미지 생성해주세요", "이미지 생성", "그려줘", "그림", "image generation", "generate"]
        for word in remove_words:
            image_prompt = image_prompt.replace(word, "").strip()
        
        if not image_prompt:
            return "🎨 **이미지 생성**\n\n생성하고 싶은 이미지에 대한 설명을 입력해주세요.\n\n예시:\n• '아름다운 석양과 바다'\n• '귀여운 고양이가 놀고 있는 모습'\n• 'A beautiful sunset over the ocean'"
        
        # 향후 Stable Diffusion이나 DALL-E 등의 이미지 생성 모델 연동 예정
        return f"🎨 **이미지 생성**\n\n요청된 이미지: '{image_prompt}'\n\n⚠️ 현재는 기본 응답을 제공하고 있습니다.\n향후 AI 이미지 생성 모델(Stable Diffusion 등)이 추가될 예정입니다.\n\n💡 **준비 중인 기능:**\n• 텍스트 → 이미지 변환\n• 다양한 스타일 지원\n• 고품질 이미지 생성"
        
    except Exception as e:
        print(f"[❗이미지 생성 오류] {str(e)}")
        return "❌ 이미지 생성 처리 중 오류가 발생했습니다."


def handle_general_search(user_input, user_email, app_password):
    """일반 키워드 메일 검색 (개선된 버전)"""
    try:
        print(f"[🔍 일반 검색 시작] 입력: '{user_input}', 사용자: {email}")
        
        # 검색 키워드 추출 개선
        search_keywords = user_input.lower()
        
        # 불필요한 단어들 제거
        remove_words = ["찾아줘", "찾아주세요", "검색해줘", "검색", "find", "search", "메일", "이메일", "email"]
        for word in remove_words:
            search_keywords = search_keywords.replace(word, "").strip()
        
        print(f"[🎯 추출된 키워드] '{search_keywords}'")
        
        if not search_keywords:
            return "🔍 **메일 검색**\n\n검색하고 싶은 키워드를 입력해주세요.\n\n예시:\n• '회의 관련 메일 찾아줘'\n• '프로젝트 업데이트 검색'\n• '급한 메일 찾기'"
        
        # 실제 메일 검색 로직
        try:
            # 메일 서버 연결
            print("[📧 메일 서버 연결 시작]")
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(user_email, app_password)
            mail.select("inbox")
            print("[✅ 메일 서버 연결 성공]")
            
            # 더 많은 메일 검색 (범위 확대)
            N = 50  # 50개로 증가
            status, data_result = mail.search(None, "ALL")
            all_mail_ids = data_result[0].split()
            mail_ids = all_mail_ids[-N:]
            
            print(f"[📊 검색 범위] 총 {len(all_mail_ids)}개 중 최근 {len(mail_ids)}개 검색")
            
            found_emails = []
            processed_count = 0
            
            for msg_id in mail_ids:
                try:
                    _, msg_data = mail.fetch(msg_id, "(RFC822)")
                    if not msg_data or not msg_data[0]:
                        continue
                    
                    msg = email_module.message_from_bytes(msg_data[0][1])
                    processed_count += 1
                    
                    # 제목 디코딩 (기존 summary 함수와 같은 방식)
                    raw_subject = msg.get("Subject", "")
                    try:
                        decoded_parts = decode_header(raw_subject)
                        if decoded_parts and decoded_parts[0]:
                            decoded_subject = decoded_parts[0]
                            subject_bytes = decoded_subject[0]
                            subject_encoding = decoded_subject[1]
                            
                            if isinstance(subject_bytes, bytes):
                                if subject_encoding is None:
                                    subject_encoding = 'utf-8'
                                try:
                                    subject = subject_bytes.decode(subject_encoding)
                                except (UnicodeDecodeError, LookupError):
                                    subject = subject_bytes.decode('utf-8', errors='ignore')
                            else:
                                subject = str(subject_bytes)
                        else:
                            subject = "(제목 없음)"
                    except Exception as e:
                        subject = raw_subject if raw_subject else "(제목 없음)"
                    
                    # 발신자 정보
                    name, addr = parseaddr(msg.get("From"))
                    from_field = f"{name} <{addr}>" if name else addr
                    
                    # 날짜 정보
                    raw_date = msg.get("Date", "")
                    try:
                        date_obj = parsedate_to_datetime(raw_date)
                        date_str = date_obj.strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = raw_date[:16] if len(raw_date) >= 16 else raw_date
                    
                    # 본문 추출 (간단한 버전)
                    body = ""
                    try:
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                                    charset = part.get_content_charset() or "utf-8"
                                    body += part.get_payload(decode=True).decode(charset, errors="ignore")
                        else:
                            charset = msg.get_content_charset() or "utf-8"
                            body = msg.get_payload(decode=True).decode(charset, errors="ignore")
                        body = body.strip()[:200]  # 처음 200자만
                    except Exception as e:
                        body = ""
                    
                    # 개선된 키워드 검색 (제목, 발신자, 본문에서 모두 검색)
                    search_in = f"{subject} {from_field} {body}".lower()
                    
                    # 여러 키워드 중 하나라도 매칭되면 포함
                    keywords = search_keywords.split()
                    if any(keyword in search_in for keyword in keywords):
                        found_emails.append({
                            "subject": subject[:60] + "..." if len(subject) > 60 else subject,
                            "from": from_field[:40] + "..." if len(from_field) > 40 else from_field,
                            "date": date_str,
                            "preview": body[:100] + "..." if len(body) > 100 else body
                        })
                        
                        print(f"[✅ 매칭] {subject[:30]}...")
                        
                        if len(found_emails) >= 8:  # 최대 8개까지
                            break
                            
                except Exception as e:
                    print(f"[⚠️ 메일 처리 오류] {str(e)}")
                    continue
            
            mail.close()
            mail.logout()
            
            print(f"[📊 검색 완료] {processed_count}개 처리, {len(found_emails)}개 발견")
            
            if found_emails:
                result = f"🔍 **검색 결과**\n\n키워드: '{search_keywords}'\n검색된 메일: {len(found_emails)}개 (총 {processed_count}개 중)\n\n"
                for i, mail_info in enumerate(found_emails, 1):
                    result += f"**{i}. {mail_info['subject']}**\n"
                    result += f"📤 {mail_info['from']}\n"
                    result += f"📅 {mail_info['date']}\n"
                    if mail_info['preview']:
                        result += f"💬 {mail_info['preview']}\n"
                    result += "\n"
                result += "💡 더 정확한 검색을 위해 구체적인 키워드를 사용해보세요."
                return result
            else:
                return f"🔍 **검색 결과**\n\n키워드: '{search_keywords}'\n검색 범위: 최근 {processed_count}개 메일\n\n❌ 관련된 메일을 찾을 수 없습니다.\n\n💡 **검색 팁:**\n• 다른 키워드로 시도해보세요\n• 발신자 이름이나 회사명 사용\n• 메일 제목의 핵심 단어 사용\n• 영어/한국어 모두 시도"
                
        except Exception as e:
            print(f"[❗메일 검색 오류] {str(e)}")
            return f"❌ 메일 검색 중 오류가 발생했습니다.\n\n오류 내용: {str(e)}\n\n💡 로그인 정보나 네트워크 연결을 확인해주세요."
        
    except Exception as e:
        print(f"[❗일반 검색 오류] {str(e)}")
        return "❌ 검색 처리 중 오류가 발생했습니다."



# 2. handle_person_search 함수도 개선

def handle_person_search(user_input, user_email, app_password):
    """특정 사람 메일 검색 (개선된 버전)"""
    try:
        print(f"[👤 사람 검색 시작] 입력: '{user_input}'")
        
        # Qwen을 이용해 사람 이름/이메일 추출
        search_target = extract_search_target_with_qwen(user_input)
        print(f"[🎯 추출된 대상] '{search_target}'")
        
        # Qwen 실패 시 간단한 추출 방법
        if not search_target or len(search_target.strip()) < 2:
            # 간단한 이름/이메일 추출
            words = user_input.split()
            potential_targets = []
            
            for word in words:
                # 이메일 주소 패턴
                if "@" in word and "." in word:
                    potential_targets.append(word)
                # 한국어 이름 패턴 (2-4글자)
                elif len(word) >= 2 and len(word) <= 4 and word.replace(" ", "").isalpha():
                    potential_targets.append(word)
            
            if potential_targets:
                search_target = potential_targets[0]
            else:
                return "👤 **사람별 메일 검색**\n\n찾고 싶은 사람의 이름이나 이메일 주소를 명확히 알려주세요.\n\n예시:\n• '김철수님의 메일'\n• 'john@company.com 메일'\n• '홍길동 교수님 메일'"
        
        print(f"[🔍 최종 검색 대상] '{search_target}'")
        
        try:
            # 메일 서버 연결
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(user_email, app_password)
            mail.select("inbox")
            
            # 더 많은 메일 검색
            N = 100  # 100개로 증가
            status, data_result = mail.search(None, "ALL")
            all_mail_ids = data_result[0].split()
            mail_ids = all_mail_ids[-N:]
            
            print(f"[📊 검색 범위] 최근 {len(mail_ids)}개 메일에서 검색")
            
            found_emails = []
            processed_count = 0
            
            for msg_id in mail_ids:
                try:
                    _, msg_data = mail.fetch(msg_id, "(RFC822)")
                    if not msg_data or not msg_data[0]:
                        continue
                    
                    msg = email_module.message_from_bytes(msg_data[0][1])
                    processed_count += 1
                    
                    # 발신자 정보 추출
                    from_header = msg.get("From", "")
                    name, addr = parseaddr(from_header)
                    from_field = f"{name} <{addr}>" if name else addr
                    
                    # 제목 추출 (간단한 방법)
                    subject = str(msg.get("Subject", ""))[:80]
                    
                    # 날짜 추출
                    date_field = str(msg.get("Date", ""))[:25]
                    
                    # 검색 대상이 발신자 정보에 포함되는지 확인 (대소문자 무시, 부분 매칭)
                    search_lower = search_target.lower()
                    from_lower = from_field.lower()
                    
                    # 더 관대한 매칭
                    if (search_lower in from_lower or 
                        any(part.strip() in from_lower for part in search_lower.split() if part.strip()) or
                        (len(search_lower) >= 3 and search_lower in from_lower.replace(" ", ""))):
                        
                        found_emails.append({
                            "subject": subject,
                            "from": from_field,
                            "date": date_field
                        })
                        
                        print(f"[✅ 매칭] {from_field} -> {subject[:30]}...")
                        
                        if len(found_emails) >= 10:  # 최대 10개까지
                            break
                            
                except Exception as e:
                    continue
            
            mail.close()
            mail.logout()
            
            print(f"[📊 사람 검색 완료] {processed_count}개 처리, {len(found_emails)}개 발견")
            
            if found_emails:
                result = f"👤 **사람별 메일 검색 결과**\n\n검색 대상: '{search_target}'\n발견된 메일: {len(found_emails)}개 (총 {processed_count}개 중)\n\n"
                for i, mail_info in enumerate(found_emails, 1):
                    result += f"**{i}. {mail_info['subject']}**\n"
                    result += f"📤 {mail_info['from']}\n"
                    result += f"📅 {mail_info['date']}\n\n"
                result += "💡 특정 메일을 자세히 보려면 메일 리스트에서 확인하세요."
                return result
            else:
                return f"👤 **사람별 메일 검색 결과**\n\n검색 대상: '{search_target}'\n검색 범위: 최근 {processed_count}개 메일\n\n❌ 해당 사람의 메일을 찾을 수 없습니다.\n\n💡 **검색 팁:**\n• 정확한 이름 사용: '{search_target}' → 다른 표기법 시도\n• 이메일 주소로 시도\n• 성이나 이름만으로 시도\n• 영문/한글 이름 모두 시도"
                
        except Exception as e:
            print(f"[❗사람 검색 오류] {str(e)}")
            return f"❌ 사람별 메일 검색 중 오류가 발생했습니다.\n\n오류: {str(e)}"
        
    except Exception as e:
        print(f"[❗사람 검색 핸들러 오류] {str(e)}")
        return "❌ 사람 검색 처리 중 오류가 발생했습니다."


@app.route('/api/test', methods=['POST'])
def test():
    data = request.get_json()
    text = data.get("text", "")
    email = data.get("email", "")
    
    user_key = get_user_key(email) if email else "anonymous"
    
    return jsonify({
        "message": f"✅ 백엔드 정상 작동: {text[:20]}...",
        "user_session": user_key[:8] + "..." if email else "no_session"
    })

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

        # 사용자 세션 확인
        user_key = get_user_key(sender_email)
        if user_key not in user_sessions:
            return jsonify({"error": "로그인이 필요합니다."}), 401

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()

        print(f"[📤 메일 전송 성공] 사용자: {sender_email}, 수신자: {to}")

        return jsonify({"message": "✅ 메일 전송 성공"}), 200

    except Exception as e:
        print("[❗메일 전송 실패]", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/session-info', methods=['GET'])
def session_info():
    """현재 활성 세션 정보 반환 (디버그용)"""
    return jsonify({
        "active_sessions": len(user_sessions),
        "session_keys": [key[:8] + "..." for key in user_sessions.keys()]
    })

@app.route('/', methods=['GET'])
def health_check():
    return "✅ 백엔드 정상 작동 중 (사용자 세션 분리 적용)", 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
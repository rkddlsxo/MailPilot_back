# MailPilot AI - Backend API
AI 기반 이메일 관리 서버 (Flask 기반)

로컬 PC 또는 서버에서 실행되는 오픈소스 Flask API 서버로, 이메일 요약, 분류, 검색, AI 답장 생성 및 챗봇 기능을 제공합니다. 클라이언트는 Electron 기반 데스크탑 앱 또는 React 웹에서 요청을 전송합니다.

---

# 팀 구성원

| 이름 | 이메일                 |퀄컴ID |
|------|-----------------------|--------------------------------|
|최수운|csw21c915@gmail.com     |csw21c915@gmail.com             |
|강인태|rkddlsxo12345@naver.com |rkddlsxo12345@naver.com         |
|김관영|kwandol02@naver.com     |kwandol02@naver.com             |
|김진성|jinsung030405@gmail.com |jinsung030405@gmail.com         |
|이상민|haleeho2@naver.com      |haleeho2@naver.com              |

---

## 기술 스택

### Backend (Python Flask API)
- **Flask**: RESTful API 서버
- **Transformers**: Hugging Face 모델 (BART, Qwen)
- **Nomic**: 임베딩 및 분류
- **scikit-learn**: 코사인 유사도 계산
- **imaplib/smtplib**: Gmail 연동

### Frontend (Electron Desktop App)
- **Electron**: 크로스 플랫폼 데스크탑 앱 프레임워크
- **HTML/CSS/JavaScript**: 웹 기반 UI


---

# 주요 AI 기능 설명
1. 이메일 요약 및 분류 (/api/summary)
최근 N개 메일을 가져와 본문 요약 (BART 사용)

내용 기반으로 카테고리 자동 분류 (Nomic Embedding + Cosine Similarity)

중요, 스팸, 일반 태그도 추가

2. AI 답장 생성 (/api/generate-ai-reply)
수신 메일 내용에 기반하여 자동으로 영어 답장을 생성

Hugging Face Qwen 모델 사용 (Qwen2.5-7B-Instruct)

간결하고 정중한 형식으로 작성

3. 검색 기능 (/api/email-search)
자연어 입력 → Qwen 모델로 대상 추출 → 최근 메일 검색

발신자 이름/주소, 제목, 본문 내용 등 다중 필드 검색

4. 챗봇 인터페이스 (/api/chatbot)
사용자의 요청을 4가지 의도 중 분류:

문법/맞춤법 교정

텍스트 기반 이미지 생성 (준비 중)

일반 메일 검색

특정 사람 메일 검색

분류는 Nomic 임베딩 + 코사인 유사도로 판단

---

📡 주요 API 엔드포인트

| 경로                       | 메서드  | 설명                     |
| ------------------------ | ---- | ---------------------- |
| `/api/login`             | POST | 사용자 로그인 및 세션 생성        |
| `/api/logout`            | POST | 사용자 로그아웃 및 세션 종료       |
| `/api/summary`           | POST | 이메일 요약 및 자동 분류         |
| `/api/generate-ai-reply` | POST | 수신 메일에 대한 AI 답장 생성     |
| `/api/email-search`      | POST | 키워드/사람 기반 이메일 검색       |
| `/api/chatbot`           | POST | 챗봇 인터페이스 (문법 교정, 검색 등) |
| `/api/send`              | POST | Gmail을 통한 이메일 전송       |
| `/api/session-info`      | GET  | 디버그용: 현재 세션 정보 확인      |
| `/`                      | GET  | 서버 상태 확인 (헬스 체크)       |

---

📦 설치 및 실행 방법

프론트엔드 설치 및 실행 방법은 다음 저장소에서 확인하세요:

**🔗 [MailPilot 프론트엔드 저장소]([copilot_project](https://github.com/jinsunghub/copilot_project))**

🔐 사전 준비 사항
Gmail 2단계 인증 필수

앱 비밀번호 생성 (Google 계정 설정)

.env 또는 시스템 환경 변수에 Hugging Face 토큰 설정: HF_TOKEN


## 1. 레포지토리 클론
git clone https://github.com/rkddlsxo/MailPilot_back.git
cd MailPilot_back

## 2. 의존성 설치
pip install -r requirements.txt

## 3. 실행
python app.py

서버는 기본적으로 http://localhost:5001 에서 실행됩니다.

## 4. Nomic/Qwen 모델을 구동하기 위한 패키지 설치

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes nomic scikit-learn qai-hub einops safetensors

## 5. 필요한 라이브러리 설치

!pip install ultralytics
!pip install opencv-python-headless
!pip install pillow
!pip install pandas
!pip install numpy

## 6. 실행/사용 방법

### 프로젝트 디렉토리로 이동
cd MailPilot_back

### 가상환경(optional) 설정 후 실행
python app.py


---

🔐 보안
❗ 절대 일반 Gmail 비밀번호를 사용하지 마세요

반드시 앱 비밀번호를 생성하여 사용해야 합니다

Gmail 계정은 2단계 인증이 활성화되어 있어야 앱 비밀번호를 생성할 수 있습니다

서버 내 환경 변수(HF_TOKEN) 등 민감한 정보는 .env 파일이나 환경 설정에 별도 관리 권장

🖥 시스템 요구사항
백엔드 API 서버가 먼저 실행되어 있어야 프론트엔드(Electron 앱 또는 웹)에서 정상 연결됩니다

인터넷 연결 필수

Gmail 서버(IMAP/SMTP) 연동

AI 모델 호출 (Hugging Face Inference API, Nomic API 등)

포트 5001이 방화벽이나 보안 소프트웨어에 의해 차단되지 않아야 합니다

로컬 머신에서 실행 시 Python 3.8 이상 권장

---
### MIT 라이선스

```
MIT License

Copyright (c) 2024 MailPilot AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 기타 오픈소스 라이선스

이 프로젝트는 다음 오픈소스 라이브러리들을 사용합니다:
자세한 프론트엔드 의존성 및 라이선스 정보는 [프론트엔드 저장소]([copilot_project](https://github.com/jinsunghub/copilot_project))를 참조하세요:
### Frontend Dependencies
- **Electron**: MIT License
- **Bootstrap**: MIT License
- **Font Awesome**: Font Awesome Free License

### Backend Dependencies (API 서버)
- **Flask**: BSD License
- **Transformers (Hugging Face)**: Apache License 2.0
- **PyTorch**: BSD License
- **scikit-learn**: BSD License
- **Nomic**: Proprietary License (API 서비스)

각 라이브러리의 전체 라이선스 텍스트는 해당 프로젝트의 공식 저장소에서 확인할 수 있습니다

---

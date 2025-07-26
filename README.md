# E.M.Pilot - Backend

**AI 기반 이메일 관리 데스크탑 앱 (Tauri 기반)**

> 로컬 PC NPU를 활용한, 오픈소스 on-device 대화형 AI 이메일 클라이언트

---

## 응용 프로그램에 대한 설명

E.M.Pilot는 Gmail 계정과 연동하여 이메일을 자동으로 분류, 요약하고 AI 기반 답장을 생성하고, 이메일 사용자들이 지금까지 활용하지 못했던 기능이나, 활용에 편의를 더할 기능을 대화형 인터페이스를 활용하여 제공해주는 스마트 이메일 관리 데스크탑 애플리케이션입니다. React, Flask 프레임워크를 기반으로 Tauri을 활용하여 개발한 데스크탑 앱으로, 로컬 PC의 NPU를 활용하여 AI 모델을 실행하여 클라우드 의존성을 최소화했습니다.

### AI 모델을 활용한 주요 기능

| 기능                         | 설명                                      |
| ---------------------------- | ----------------------------------------- |
| 스팸/중요/보낸/내게쓴/필터링  | 탭 별로 메일을 자동 분류하여 확인 가능    |
| 메일 요약 보기               | 리스트에서 메일 내용을 요약으로 미리 확인 |
| 보낸 사람 검색 기능          | 보낸 사람 기준 해당 메일 필터링           |
| To-do 표시                  | 사용자의 주요 일정을 자동으로 정리하여 제공 |
| 데스크탑 앱                  | Tauri 기반의 독립 실행형 앱 구성       |
| AI 답장 생성                 | 수신된 이메일에 대한 자동 답장 생성       |
| 대화형 인퍼페이스            | 문법 교정, 캘린더 생성, 검색 기능 요청  |

---

## 팀 구성원

| 이름   | 이메일                     | 퀄컴ID                     |
|--------|----------------------------|----------------------------|
| 최수운 | csw21c915@gmail.com        | csw21c915@gmail.com        |
| 강인태 | rkddlsxo12345@naver.com    | rkddlsxo12345@naver.com    |
| 김관영 | kwandol02@naver.com        | kwandol02@naver.com        |
| 김진성 | jinsung030405@gmail.com    | jinsung030405@gmail.com    |
| 이상민 | haleeho2@naver.com         | haleeho2@naver.com         |

---

## 기술 스택

### Backend (Python Flask API)
- **Flask**: RESTful API 서버
- **Transformers**: Hugging Face 모델 (BART, Qwen)
- **Nomic**: 임베딩 및 분류
- **scikit-learn**: 코사인 유사도 계산
- **imaplib/smtplib**: Gmail 연동

### Frontend (Tauri Desktop App)
- **Tauri**: 크로스 플랫폼 데스크탑 앱 프레임워크
- **HTML/CSS/JavaScript**: 웹 기반 UI

### 주요 API 엔드포인트

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

## 응용 프로그램 설치 방법

### 1. 레포지토리 클론
```bash
git clone https://github.com/rkddlsxo/MailPilot_back.git
cd MailPilot_back
```

### 2. 가상환경 설정 및 실행
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 의존성 패키지 다운로드
```bash
pip install flask flask-cors
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes scikit-learn qai-hub einops safetensors nomic
pip install ultralytics opencv-python-headless pillow pandas numpy
```

### 4. 실행
```bash
python app.py
```

서버는 기본적으로 http://localhost:5001 에서 실행됩니다.

### 5. 프론트엔드 설치 및 실행
프론트엔드 설치 및 실행 방법은 다음 저장소에서 확인하세요:

**🔗 [MailPilot 프론트엔드 저장소](https://github.com/jinsunghub/copilot_project)**

---

## 실행/사용 방법

### 프로젝트 디렉토리로 이동 후 가상환경 설정 후 실행
```bash
cd MailPilot_back
python app.py
```

### 로그인
1. 데스크탑 앱에서 Gmail 주소 입력
2. Gmail 앱 비밀번호 입력 (일반 비밀번호 아님!)
3. 로그인 버튼 클릭

### 이메일 관리
- **새로고침** 버튼으로 최근 이메일 가져오기
- 탭별로 자동 분류된 이메일 확인 (스팸/중요/보낸함 등)
- 이메일 리스트에서 자동 생성된 요약 확인
- 대화형 인터페이스를 활용하여 원하는 기능 사용 가능

### AI 기능 활용
- **답장 생성**: 이메일 선택 후 "AI 답장" 버튼
- **요약 및 분류**: 자동으로 이메일 요약 및 분류 내용 제공
- **챗봇**: 맞춤법 교정, 메일 찾기 등

---

## ⚠️ 주의사항

### 보안
- **절대 일반 Gmail 비밀번호를 사용하지 마세요**
- 반드시 앱 비밀번호를 생성하여 사용
- Gmail 2단계 인증이 활성화되어 있어야 함

### 시스템 요구사항
- 백엔드 API 서버가 먼저 실행되어 있어야 함
- 인터넷 연결 필수 (Gmail 접속 및 AI 모델 사용)

---

## 라이선스

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

### 기타 오픈소스 라이선스

이 프로젝트는 다음 오픈소스 라이브러리들을 사용합니다:

**Frontend Dependencies**
- **Tauri**: MIT License
- **Bootstrap**: MIT License
- **Font Awesome**: Font Awesome Free License

**Backend Dependencies (API 서버)**

자세한 백엔드 의존성 및 라이선스 정보는 [백엔드 저장소](https://github.com/rkddlsxo/MailPilot_back.git)를 참조하세요:
- **Flask**: BSD License
- **Transformers (Hugging Face)**: Apache License 2.0
- **PyTorch**: BSD License
- **scikit-learn**: BSD License
- **Nomic**: Proprietary License (API 서비스)

각 라이브러리의 전체 라이선스 텍스트는 해당 프로젝트의 공식 저장소에서 확인할 수 있습니다.

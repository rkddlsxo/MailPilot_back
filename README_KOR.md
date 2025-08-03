# E.M.Pilot

**AI 기반 이메일 관리 데스크탑 앱**

> 로컬 PC NPU를 활용하여, 오픈소스 on-device 대화형 AI 이메일 클라이언트 데스크탑 앱

---

## 응용 프로그램에 대한 설명

MailPilot AI는 Gmail 계정과 연동하여 이메일을 자동으로 분류 및 요약, AI 기반 자동 답장 생성, 기타 기능들을 활용하여, 지금까지 이메일 사용자들이 활용하지 못했던 기능이나, 이메일 활용에 편의를 더할 기능을 추가하여, 대화형 인터페이스를 통해 해당 기능들을 제공해주는 AI 이메일 관리 애플리케이션입니다. 
React, Flask 프레임워크를 기반으로 Tauri를 활용하여 개발한 데스크탑 앱으로, 퀄컴의 copilot+ pc의 NPU를 활용하여 AI 모델을 실행하여, 기존 클라우드를 활용하는 환경의 의존성을 최소화했습니다.

### AI 모델을 활용한 주요 기능

| 기능                         | 설명                                      |
| ---------------------------- | ----------------------------------------- |
| 스팸/중요/보낸/내게쓴/필터링  | 탭 별로 메일을 자동 분류하여 확인 가능    |
| 메일 요약 보기               | 리스트에서 메일 내용을 요약으로 미리 확인 |
| 보낸 사람 검색 기능          | 보낸 사람 기준 해당 메일 필터링           |
| To-do(할인 관리)표시         | 사용자의 주요 일정을 자동으로 정리하여 제공 |
| 첨부파일 요약 기능            | 첨부파일의 이미지 및 문서 내용 자동 요약    |
| AI 답장 자동 생성            | 수신된 이메일에 대한 자동 답장 생성       |
| 대화형 인퍼페이스            | 문법 교정, 캘린더 내용 추가, 검색 기능 등 이메일 관리에 필요한 기능 요청  |

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

### Backend
- **Flask**: 백엔드 서버
- **Transformers**: Hugging Face 모델 -> 퀄컴 ai 허브를 활용하여, 로컬 pc의 모델로 수정 예정
- **AI Model**: Nomic, QWEN LLM, EASY_OCR 모델 사용 중

## 응용 프로그램 설치 및 실행 방법

### 1. 프로젝트 클론
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
pip install -r requirements.txt
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

### 로그인
1. 데스크탑 앱에서 Gmail 주소 입력
2. Gmail 앱 비밀번호 입력 (일반 비밀번호 아님!)
3. 로그인 버튼 클릭

### 이메일 관리
- '새로고침' 버튼으로 최근 이메일 가져오기
- 탭별로 분류된 이메일 및 할일 관리 확인 (스팸/중요/보낸함 등)
- 이메일 리스트에서 요약 내용 확인
- 대화형 인터페이스를 활용하여 원하는 기능 사용 가능

### AI 기능 활용
- **답장 생성**: 이메일 선택 후 "AI 답장" 버튼
- **요약 및 분류**: 자동으로 이메일 내용 및 첨부파일 요약, 이메일 키워드 분류 내용 제공
- **챗봇**: 맞춤법 교정, 메일 찾기, 할일 추가, 검색 등 이메일 사용자들에게 다양한 기능 제공



---

## ⚠️ 주의사항

### 보안
- **절대 일반 Gmail 비밀번호를 사용하지 마세요**
- 반드시 앱 비밀번호를 생성하여 사용
- Gmail 2단계 인증이 활성화되어 있어야 함

### 시스템 요구사항
- 백엔드 API 서버가 먼저 실행되어 있어야 함
- 인터넷 연결 필수 (Gmail 접속 및 AI 모델 사용)

### Token 발급 필요
- 현재는 app.py에 Qalcom 디바이스가 없는 관계로, Hugging face에서 토큰을 발급받아 사용 요구
- Nomic과 Hunggface의 토큰을 발급받아서, 해당 부분 수정이 필요함
- 추후에 퀄컴 ai 허브를 활용하여, 모델 다운로드 후 사용 예정

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

**Backend**

- **Flask**: BSD License
- **Transformers (Hugging Face)**: Apache License 2.0
- **PyTorch**: BSD License
- **scikit-learn**: BSD License
- **Nomic**: Proprietary License (API 서비스)

**Frontend**

자세한 프론트엔드 라이선스 정보는 [프론트엔드 저장소](https://github.com/jinsunghub/copilot_project.git)를 참조하세요:



각 라이브러리의 전체 라이선스 텍스트는 해당 프로젝트의 공식 저장소에서 확인할 수 있습니다.

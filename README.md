# E.M.Pilot - Backend

**AI-powered Email Management Desktop App (Tauri-based)**

> Local PC NPU-powered, open-source on-device conversational AI email client

## Application Description

E.M.Pilot is a smart email management desktop application that integrates with Gmail accounts to automatically classify and summarize emails, generate AI-based replies, and provides features that email users haven't been able to utilize before or adds convenience to their usage through a conversational interface. It's a desktop app developed using React and Flask frameworks with Tauri, utilizing local PC NPU to run AI models, minimizing cloud dependency.

### Key Features Using AI Models

| Feature                          | Description                                        |
| -------------------------------- | -------------------------------------------------- |
| Spam/Important/Sent/To Me/Filter | Automatically categorize emails by tabs           |
| Email Summary View               | Preview email content summaries in the list       |
| Sender Search Function           | Filter emails by sender                            |
| To-do Display                   | Automatically organize and provide user's key schedules |
| Desktop App                      | Standalone app built with Tauri                   |
| AI Reply Generation              | Generate automatic replies to received emails     |
| Conversational Interface         | Grammar correction, calendar creation, search features |

---

## Team Members

| Name         | English Name  | Email                       | Qualcomm ID                |
|--------------|---------------|-----------------------------|----------------------------|
| Choi Sooun   | Choi Sooun    | csw21c915@gmail.com        | csw21c915@gmail.com        |
| Kang Intae   | Kang Intae    | rkddlsxo12345@naver.com    | rkddlsxo12345@naver.com    |
| Kim Kwanyoung| Kim Kwanyoung | kwandol02@naver.com        | kwandol02@naver.com        |
| Kim Jinsung  | Kim Jinsung   | jinsung030405@gmail.com    | jinsung030405@gmail.com    |
| Lee Sangmin  | Lee Sangmin   | haleeho2@naver.com         | haleeho2@naver.com         |

---

## Technology Stack

### Backend (Python Flask API)
- **Flask**: RESTful API server
- **Transformers**: Hugging Face models (BART, Qwen)
- **Nomic**: Embedding and classification
- **scikit-learn**: Cosine similarity calculation
- **imaplib/smtplib**: Gmail integration

### Frontend (Tauri Desktop App)
- **Tauri**: Cross-platform desktop app framework
- **HTML/CSS/JavaScript**: Web-based UI

### Main API Endpoints

| Route                    | Method | Description                          |
| ------------------------ | ------ | ------------------------------------ |
| `/api/login`             | POST   | User login and session creation      |
| `/api/logout`            | POST   | User logout and session termination |
| `/api/summary`           | POST   | Email summarization and auto-classification |
| `/api/generate-ai-reply` | POST   | AI reply generation for received emails |
| `/api/email-search`      | POST   | Keyword/person-based email search    |
| `/api/chatbot`           | POST   | Chatbot interface (grammar correction, search, etc.) |
| `/api/send`              | POST   | Send emails through Gmail            |
| `/api/session-info`      | GET    | Debug: Check current session info    |
| `/`                      | GET    | Server status check (health check)   |

---

## Application Installation Guide

### 1. Clone Repository
```bash
git clone https://github.com/rkddlsxo/MailPilot_back.git
cd MailPilot_back
```

### 2. Set up and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install flask flask-cors
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes scikit-learn qai-hub einops safetensors nomic
pip install ultralytics opencv-python pillow pandas numpy
```

### 4. Run
```bash
python app.py
```

The server runs by default at http://localhost:5001.

### 5. Frontend Installation and Execution
For frontend installation and execution instructions, check the following repository:

**🔗 [MailPilot Frontend Repository](https://github.com/jinsunghub/copilot_project)**

---

## Execution/Usage Instructions

### Navigate to Project Directory, Set up Virtual Environment, and Run
```bash
cd MailPilot_back
python app.py
```

### Login
1. Enter Gmail address in the desktop app
2. Enter Gmail app password (not your regular password!)
3. Click login button

### Email Management
- Use **Refresh** button to fetch recent emails
- Check automatically categorized emails by tabs (Spam/Important/Sent, etc.)
- View auto-generated summaries in email list
- Use conversational interface for desired features

### AI Feature Utilization
- **Reply Generation**: Select email and click "AI Reply" button
- **Summary and Classification**: Automatically provides email summary and classification content
- **Chatbot**: Grammar correction, email search, etc.

---

## ⚠️ Important Notes

### Security
- **Never use your regular Gmail password**
- Must create and use an app password
- Gmail 2-step verification must be enabled

### System Requirements
- Backend API server must be running first
- Internet connection required (for Gmail access and AI model usage)

### Token Generation Required
- Currently using Hugging Face tokens in app.py due to lack of Qualcomm device
- Need to obtain tokens from Nomic and Hugging Face and modify the corresponding sections in app.py file

---

## License

### MIT License

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

### Other Open Source Licenses

This project uses the following open source libraries:

**Frontend Dependencies**
- **Tauri**: MIT License
- **Bootstrap**: MIT License
- **Font Awesome**: Font Awesome Free License

**Backend Dependencies (API Server)**

For detailed backend dependencies and license information, refer to the [backend repository](https://github.com/rkddlsxo/MailPilot_back.git):
- **Flask**: BSD License
- **Transformers (Hugging Face)**: Apache License 2.0
- **PyTorch**: BSD License
- **scikit-learn**: BSD License
- **Nomic**: Proprietary License (API service)

Full license text for each library can be found in their respective official repositories.

### Key Features Using AI Models

| Feature                          | Description                                        |
| -------------------------------- | -------------------------------------------------- |
| Spam/Important/Sent/To Me/Filter | Automatically categorize emails by tabs           |
| Email Summary View               | Preview email content summaries in the list       |
| Sender Search Function           | Filter emails by sender                            |
| To-do Display                   | Automatically organize and provide user's key schedules |
| Desktop App                      | Standalone app built with Tauri                   |
| AI Reply Generation              | Generate automatic replies to received emails     |
| Conversational Interface         | Grammar correction, calendar creation, search features |

---

## Team Members

| Name         | English Name  | Email                       | Qualcomm ID                |
|--------------|---------------|-----------------------------|----------------------------|
| Choi Sooun   | Choi Sooun    | csw21c915@gmail.com        | csw21c915@gmail.com        |
| Kang Intae   | Kang Intae    | rkddlsxo12345@naver.com    | rkddlsxo12345@naver.com    |
| Kim Kwanyoung| Kim Kwanyoung | kwandol02@naver.com        | kwandol02@naver.com        |
| Kim Jinsung  | Kim Jinsung   | jinsung030405@gmail.com    | jinsung030405@gmail.com    |
| Lee Sangmin  | Lee Sangmin   | haleeho2@naver.com         | haleeho2@naver.com         |

---

## Technology Stack

### Backend (Python Flask API)
- **Flask**: RESTful API server
- **Transformers**: Hugging Face models (BART, Qwen)
- **Nomic**: Embedding and classification
- **scikit-learn**: Cosine similarity calculation
- **imaplib/smtplib**: Gmail integration

### Frontend (Tauri Desktop App)
- **Tauri**: Cross-platform desktop app framework
- **HTML/CSS/JavaScript**: Web-based UI

### Main API Endpoints

| Route                    | Method | Description                          |
| ------------------------ | ------ | ------------------------------------ |
| `/api/login`             | POST   | User login and session creation      |
| `/api/logout`            | POST   | User logout and session termination |
| `/api/summary`           | POST   | Email summarization and auto-classification |
| `/api/generate-ai-reply` | POST   | AI reply generation for received emails |
| `/api/email-search`      | POST   | Keyword/person-based email search    |
| `/api/chatbot`           | POST   | Chatbot interface (grammar correction, search, etc.) |
| `/api/send`              | POST   | Send emails through Gmail            |
| `/api/session-info`      | GET    | Debug: Check current session info    |
| `/`                      | GET    | Server status check (health check)   |

---

## Application Installation Guide

### 1. Clone Repository
```bash
git clone https://github.com/rkddlsxo/MailPilot_back.git
cd MailPilot_back
```

### 2. Set up and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install flask flask-cors
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes scikit-learn qai-hub einops safetensors nomic
pip install ultralytics opencv-python pillow pandas numpy
```

### 4. Run
```bash
python app.py
```

The server runs by default at http://localhost:5001.

### 5. Frontend Installation and Execution
For frontend installation and execution instructions, check the following repository:

**🔗 [MailPilot Frontend Repository](https://github.com/jinsunghub/copilot_project)**

---

## Execution/Usage Instructions

### Navigate to Project Directory, Set up Virtual Environment, and Run
```bash
cd MailPilot_back
python app.py
```

### Login
1. Enter Gmail address in the desktop app
2. Enter Gmail app password (not your regular password!)
3. Click login button

### Email Management
- Use **Refresh** button to fetch recent emails
- Check automatically categorized emails by tabs (Spam/Important/Sent, etc.)
- View auto-generated summaries in email list
- Use conversational interface for desired features

### AI Feature Utilization
- **Reply Generation**: Select email and click "AI Reply" button
- **Summary and Classification**: Automatically provides email summary and classification content
- **Chatbot**: Grammar correction, email search, etc.

---

## ⚠️ Important Notes

### Security
- **Never use your regular Gmail password**
- Must create and use an app password
- Gmail 2-step verification must be enabled

### System Requirements
- Backend API server must be running first
- Internet connection required (for Gmail access and AI model usage)

### Token Generation Required
- Currently using Hugging Face tokens in app.py due to lack of Qualcomm device
- Need to obtain tokens from Nomic and Hugging Face and modify the corresponding sections in app.py file

---

## License

### MIT License

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

### Other Open Source Licenses

This project uses the following open source libraries:

**Frontend Dependencies**
- **Tauri**: MIT License
- **Bootstrap**: MIT License
- **Font Awesome**: Font Awesome Free License

**Backend Dependencies (API Server)**

For detailed backend dependencies and license information, refer to the [backend repository](https://github.com/rkddlsxo/MailPilot_back.git):
- **Flask**: BSD License
- **Transformers (Hugging Face)**: Apache License 2.0
- **PyTorch**: BSD License
- **scikit-learn**: BSD License
- **Nomic**: Proprietary License (API service)

Full license text for each library can be found in their respective official repositories.

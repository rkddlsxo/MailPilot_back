📄 If you need the Korean version of this README, please see **README_KOR.md**.

# E.M.Pilot

**AI Email Management Desktop App**

> An open-source on-device conversational AI email client desktop application that leverages your local PC's NPU

---

## Application Description

MailPilot AI is an AI email management application that integrates with Gmail accounts to automatically classify and summarize emails, generate AI-based auto-replies, and other features. It adds convenience to email usage by providing functions that email users have not been able to utilize before, offering these features through a conversational interface.

This desktop app was developed using React and Flask frameworks with Tauri, running AI models on Qualcomm Copilot+ PC NPUs to minimize dependency on existing cloud-based environments.

### Key Features Using AI Models

| Feature | Description |
| ------- | ----------- |
| Spam/Important/Sent/Self-sent/Filtered | Automatically classify and view emails by tabs |
| Email Summary View | Preview email content summaries in the list view |
| Sender Search Function | Filter emails by sender |
| To-Do (Task Management) Display | Automatically organize and provide user's key schedules |
| Attachment Summary Function | Automatically summarize attached images and document content |
| AI Auto-Reply Generation | Generate automatic replies for received emails |
| Conversational Interface | Grammar correction, calendar content addition, search functions, and other email management features |

---

## Team Members

| Name | Email | Qualcomm ID |
|------|-------|-------------|
| 최수운 | csw21c915@gmail.com | csw21c915@gmail.com |
| 강인태 | rkddlsxo12345@naver.com | rkddlsxo12345@naver.com |
| 김관영 | kwandol02@naver.com | kwandol02@naver.com |
| 김진성 | jinsung030405@gmail.com | jinsung030405@gmail.com |
| 이상민 | haleeho2@naver.com | haleeho2@naver.com |

---

## Tech Stack

### Backend
- **Flask**: Backend server
- **Transformers**: Hugging Face models -> will be modified to local PC models using Qualcomm AI Hub
- **AI Models**: Nomic, QWEN LLM, EASY_OCR models

## Application Installation and Execution Guide

### 1. Clone Project
```bash
git clone https://github.com/rkddlsxo/MailPilot_back.git
cd MailPilot_back
```

### 2. Set Up and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Download Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execution
```bash
python app.py
```

The server runs at http://localhost:5001 by default.

### 5. Frontend Installation and Execution
For frontend installation and execution instructions, please check the following repository:

**[MailPilot Frontend Repository](https://github.com/jinsunghub/copilot_project)**

---

## Execution/Usage Guide

### Login
1. Enter Gmail address in the desktop app
2. Enter Gmail app password (not regular password!)
3. Click login button

### Email Management
- Fetch recent emails with the 'Refresh' button
- Check emails classified by tabs and task management (Spam/Important/Sent, etc.)
- Check summary content in the email list
- Use various functions through conversational interface

### AI Feature Usage
- **Reply Generation**: Select email and click "AI Reply" button
- **Summary and Classification**: Automatically provide email content and attachment summaries, email keyword classification content
- **Chatbot**: Grammar correction, email search, task addition, search, and various other features for email users

---

## ⚠️ Important Notes

### Security
- **Never use your regular Gmail password**
- Must use app password
- Gmail 2-step verification must be activated

### System Requirements
- Backend API server must be running first
- Internet connection required (for Gmail access and AI model usage)

### Token Issuance Required
- Currently requires tokens from Hugging Face due to lack of Qualcomm devices, as specified in app.py
- Need to obtain tokens from Nomic and Hugging Face and modify accordingly
- Future plans to use Qualcomm AI hub to download models for local use

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

**Backend**

- **Flask**: BSD License
- **Transformers (Hugging Face)**: Apache License 2.0
- **PyTorch**: BSD License
- **scikit-learn**: BSD License
- **Nomic**: Proprietary License (API Service)

**Frontend**

For detailed frontend license information, please refer to the [Frontend Repository](https://github.com/jinsunghub/copilot_project.git).

The complete license text for each library can be found in the official repositories of each project.
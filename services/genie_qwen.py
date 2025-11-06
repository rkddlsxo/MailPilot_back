# -*- coding: utf-8 -*-
"""
Qwen2.5-7B (Genie · QAIRT SDK) 실행 유틸 (Snapdragon X Elite · NPU)
- 이메일 본문 요약 (<= N 단어)
- 자연어 명령에서 검색 대상(이름/이메일) 추출

실행 전 준비:
- QAIRT SDK + Qwen2.5-7B용 Genie 번들
- genie_config.json (Windows: use-mmap:false 권장, tokenizer.json/ctx-bins 경로 일치)
- genie-t2t-run.exe 및 필요한 런타임 DLL을 번들 폴더에 배치

환경변수(선택):
- GENIE_BUNDLE_DIR, GENIE_CONFIG_NAME, GENIE_EXE_NAME, GENIE_TIMEOUT_SEC
"""
from __future__ import annotations
import os
import re
import subprocess
import textwrap
from typing import Optional

# ==========================
# 설정: 경로 및 기본 파라미터
# ==========================
GENIE_BUNDLE_DIR  = os.getenv("GENIE_BUNDLE_DIR",  r"C:\Genie\Qwen2_5_7B\genie_bundle")
GENIE_CONFIG_NAME = os.getenv("GENIE_CONFIG_NAME", "genie_config.json")
GENIE_EXE_NAME    = os.getenv("GENIE_EXE_NAME",    "genie-t2t-run.exe")
GENIE_TIMEOUT_SEC = int(os.getenv("GENIE_TIMEOUT_SEC", "180"))

# ======================
# Genie (Qwen) 관련 함수
# ======================

def run_qwen_with_genie(
    prompt: str,
    bundle_dir: str = GENIE_BUNDLE_DIR,
    config_name: str = GENIE_CONFIG_NAME,
    exe_name: str = GENIE_EXE_NAME,
    timeout_sec: int = GENIE_TIMEOUT_SEC
) -> str:
    """Genie 실행기를 이용해 Qwen 프롬프트 실행 → 결과 텍스트만 추출"""
    exe_path = os.path.join(bundle_dir, exe_name)
    cfg_path = os.path.join(bundle_dir, config_name)

    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"Genie 실행 파일 없음: {exe_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Genie 설정 없음: {cfg_path}")

    prompt_path = os.path.join(bundle_dir, "__prompt_utf8.txt")
    with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(prompt)

    args = [exe_path, "-c", cfg_path, "--prompt_file", prompt_path]
    proc = subprocess.run(
        args,
        cwd=bundle_dir,
        capture_output=True,
        text=False,
        timeout=timeout_sec,
        shell=False,
    )

    stdout = (proc.stdout or b"").decode("utf-8", errors="ignore")
    stderr = (proc.stderr or b"").decode("utf-8", errors="ignore")

    # 임시 프롬프트 파일은 Windows에서 바로 삭제가 막힐 수 있어 생략(덮어쓰기 방식)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Genie 실패 (code {proc.returncode})\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}"
        )

    m = re.search(r"\[BEGIN\][^:]*:\s*(.*?)\s*\[END\]", stdout, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    tail = "\n".join([ln for ln in stdout.splitlines() if ln and not ln.startswith("[")])
    return tail.strip() or stdout.strip()

# =========================
# 프롬프트 빌더 (원본 스타일)
# =========================

def qwen_prompt_summary(email_text: str) -> str:
    system_msg = (
        "You are a helpful assistant. "
        "Write a concise, neutral summary in one or two short sentences."
    )
    user_msg = textwrap.dedent(f"""
    Summarize the following email content in <= 25 words.

    {email_text.strip()}
    """).strip()

    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
    return prompt


def qwen_prompt_extract_target(user_command: str) -> str:
    system_msg = (
        "You are an email assistant. "
        "Extract the single best name or email address the user is referring to. "
        "Respond in exactly one line in the format: The user is referring to <VALUE>"
    )
    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_command.strip()}<|im_end|>
<|im_start|>assistant
"""
    return prompt

# =========================
# 파서 & 전처리 유틸 (원본 스타일)
# =========================

def _sanitize_for_prompt(s: str) -> str:
    """프롬프트 투입 전 이메일 본문을 안전하게 소독"""
    if not s:
        return ""

    # 1. 유니코드 정규화 (스마트 쿼트 → 표준화)
    s = unicodedata.normalize("NFKC", s)

    # 2. 특수 따옴표/대시/줄임표를 ASCII로 치환
    replacements = {
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...",
        "\u00A0": " ",   # NBSP → 공백
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    # 3. 제어문자 / 제로폭 / BiDi 마커 제거
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", s)

    # 4. UTF-8 불가 바이트 제거
    s = s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    # 5. 프린트 불가능한 문자 제거 (개행/탭만 허용)
    s = "".join(ch for ch in s if ch.isprintable() or ch in "\n\t")

    # 6. 공백 정리
    s = re.sub(r"[ \t]+", " ", s)          # 연속 탭/스페이스 → 한 칸
    s = re.sub(r"\s*\n\s*", "\n", s)       # 줄바꿈 주변 공백 제거
    s = s.strip()

    return s

def _ensure_utf8(s: str) -> str:
    return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

def _sanitize_for_prompt(s: str) -> str:
    if not s:
        return ""
    s = _ensure_utf8(s)
    s = "".join(ch for ch in s if ch.isprintable() or ch in "\n\t")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s).strip()
    return s

# ======================
# 공개 API (Flask용 래퍼)
# ======================

def genie_summarize_email(email_text: str, max_words: int = 25, max_chars: int = 1500) -> str:
    """이메일 본문을 Qwen(Genie)로 요약하여 한 줄로 반환"""
    snippet = _sanitize_for_prompt(email_text[:max_chars])
    prompt = qwen_prompt_summary(snippet)
    prompt = _ensure_utf8(prompt)
    out = run_qwen_with_genie(prompt).strip()
    # 단어 수 컷(모델이 길게 답할 때 대비)
    words = out.split()
    if len(words) > max_words:
        out = " ".join(words[:max_words]).rstrip(",.;") + "..."
    return out

def genie_extract_search_target(user_command: str) -> str:
    """자연어 명령에서 '단 하나의' 대상(이름/이메일) 추출"""
    prompt = qwen_prompt_extract_target(user_command)
    prompt = _ensure_utf8(prompt)
    out = run_qwen_with_genie(prompt)
    return parse_extracted_target(out)

def genie_summarize_document(file_text: str, file_name: str, file_type: str, max_words: int = 25, max_chars: int = 1500) -> str:
    """이메일 본문을 Qwen(Genie)로 요약하여 한 줄로 반환"""
    snippet = _sanitize_for_prompt(file_text[:max_chars])
    prompt = qwen_prompt_summary_file(snippet, file_name, file_type)
    prompt = _ensure_utf8(prompt)
    out = run_qwen_with_genie(prompt).strip()
    # 단어 수 컷(모델이 길게 답할 때 대비)
    words = out.split()
    if len(words) > max_words:
        out = " ".join(words[:max_words]).rstrip(",.;") + "..."
    return out

def qwen_prompt_summary_file(file_text: str, file_name: str, file_type: str) -> str:
    system_msg = (
        "You are a helpful assistant. "
        "Write a concise, neutral summary in one or two short sentences."
    )
    user_msg = textwrap.dedent(f"""
    Below is the content of the ‘{file_name.strip()}’ file.
    Summarize the following file content in <= 25 words.
    
    file type: {file_type.strip()}
    file content:
    {file_text.strip()}
    """).strip()

    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
    return prompt

def genie_analyze_intent(user_input: str) -> str:
    """자연어 명령에서 '단 하나의' 대상(이름/이메일) 추출"""
    print("hi11111")
    prompt = qwen_prompt_command(user_input)
    print("hi22222")
    prompt = _ensure_utf8(prompt)
    print(f"[프롬프트] {prompt}")
    out = run_qwen_with_genie(prompt)
    print("hi44444")
    return parse_extracted_target_intent(out)

def parse_extracted_target_intent(text: str) -> str:
    m = re.search(r"결과:\s+(.+)$", text.strip())
    if m:
        return m.group(1).strip().rstrip(".")
    for line in text.splitlines():
        if "결과:" in line:
            return line.split("결과:", 1)[-1].strip().rstrip(".")
    return text.strip()

def qwen_prompt_command(user_input: str) -> str:
    
    system_msg = (
        "한국어 명령의 의도를 분류하세요"
        "형식 : intent|의도타입"
        "의도 타입:"
"- grammar_correction: 문법/맞춤법 교정 요청"
"- email_search: 키워드로 메일 검색"
"- person_search: 특정 사람의 메일 검색"  
"- email_statistics: 메일 개수/통계 조회"
"- settings_control: 앱 설정 변경"
"- generate_ai_reply: AI 답장 생성"
        
"예시:"
"안녕하세요 교정해주세요 → intent|grammar_correction"
"회의 관련 메일 찾아줘 → intent|email_search"
"notion team 이메일 찾아줘 → intent|email_search"
"김철수님 메일 보여줘 → intent|person_search"
"오늘 메일 몇 개? → intent|email_statistics"
"폰트 크기 18로 바꿔줘 → intent|settings_control"
"답장 생성해줘 → intent|generate_ai_reply"
    )
    prompt = f"""<|im_start|>system
    {system_msg}<|im_end|>
    <|im_start|>user
    {user_input.strip()}<|im_end|>
    <|im_start|>assistant
    """

    return prompt

def genie_reply(prompt) -> str:
    """자연어 명령에서 '단 하나의' 대상(이름/이메일) 추출"""
    prompt_renew = _ensure_utf8(prompt)
    out = run_qwen_with_genie(prompt_renew)
    return out
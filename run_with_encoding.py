#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

# UTF-8 환경변수 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# ONNX Runtime 관련 환경변수
os.environ['ORT_LOGGING_LEVEL'] = '3'  # WARNING 레벨

print("환경변수 설정 완료:")
print(f"PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING')}")
print(f"PYTHONUTF8: {os.environ.get('PYTHONUTF8')}")
print(f"LANG: {os.environ.get('LANG')}")
print(f"LC_ALL: {os.environ.get('LC_ALL')}")
print(f"ORT_LOGGING_LEVEL: {os.environ.get('ORT_LOGGING_LEVEL')}")

# 기본 인코딩 재확인
import locale
print(f"\n인코딩 확인:")
print(f"기본 인코딩: {sys.getdefaultencoding()}")
print(f"파일시스템 인코딩: {sys.getfilesystemencoding()}")
print(f"로케일: {locale.getpreferredencoding()}")

# app.py 실행
print("\n=== app.py 실행 ===")
import app
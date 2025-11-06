

# #0824 수정
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ultralytics import YOLO
import easyocr
import os
import onnxruntime as ort
import numpy as np

# ---- 추가 임포트 (서브프로세스/임시파일/직렬화 용) ----
import subprocess
import tempfile
import json
import sys
import textwrap
from pathlib import Path

# ONNX 모델 설정
USE_ONNX = True  # True: ONNX 모델 사용, False: Nomic API 사용
ONNX_MODEL_PATH = "C:/EMpilot/MailPilot_back/models/onnx_model/nomic_embed_text/model.onnx"
EASYOCR_DETECTOR_PATH = "C:/EMpilot/MailPilot_back/models/onnx_model/easyocr-easyocrdetector/model.onnx"
EASYOCR_RECOGNIZER_PATH = "C:/EMpilot/MailPilot_back/models/onnx_model/easyocr-easyocrrecognizer/model.onnx"
YOLO_ONNX_PATH = "C:/EMpilot/MailPilot_back/models/onnx_model/yolo/model.onnx"
USE_EASYOCR_ONNX = True  # True: EasyOCR ONNX 사용, False: EasyOCR API 사용
USE_YOLO_ONNX = True  # True: YOLO ONNX 사용, False: PyTorch YOLO 사용

# ----- NPU(QNN) 외부 실행에 필요한 경로(네 환경에 맞게 기본값 세팅) -----
QNN_DIR_PATH = r"C:\WoS_AI"  # ORT_QNN_Setup을 했던 루트
QNN_ONNX_MODEL_PATH = r"C:\WoS_AI\Models\nomic\model.onnx\model.onnx"  # NPU에서 쓸 nomic onnx
QNN_SETUP_PS1 = r"C:\WoS_AI\Downloads\Setup_Scripts\ort_setup.ps1"     # Activate_ORT_QNN_VENV 있는 스크립트

# Nomic API (폴백용)
try:
    from nomic import embed, login
    NOMIC_API_AVAILABLE = True
except ImportError:
    NOMIC_API_AVAILABLE = False

class AIModels:
    def __init__(self, config):
        self.config = config
        
        # 모델 인스턴스들
        self.yolo_model = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.ocr_reader = None
        self.summarizer = None
        
        # ONNX 모델 초기화
        self.onnx_session = None
        self.bert_tokenizer = None
        self.easyocr_detector_session = None
        self.easyocr_recognizer_session = None
        self.yolo_onnx_session = None
        
        # NPU 세션 추가 (완전한 NPU 파이프라인)
        self.npu_detector_session = None
        self.npu_recognizer_session = None
        self.npu_yolo_session = None
        
        # Nomic ONNX 모델 (GPU 우선, CPU 폴백)
        if USE_ONNX and os.path.exists(ONNX_MODEL_PATH):
            self.onnx_session = self._load_onnx_model(ONNX_MODEL_PATH, "Nomic 임베딩")
            if self.onnx_session:
                try:
                    # (이름은 그대로 유지) nomic 토크나이저로 교체해 정확도 맞춤
                    self.bert_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", use_fast=True)
                    self._reset_console_color()
                    print("[✅ ONNX] Nomic 모델 로딩 완료!")
                except Exception as e:
                    self._reset_console_color()
                    print(f"[❌ ONNX] Nomic 토크나이저 로딩 실패: {e}")
                    self.onnx_session = None
        
        # EasyOCR ONNX 모델 (GPU 우선, CPU 폴백)
        if USE_EASYOCR_ONNX:
            if os.path.exists(EASYOCR_DETECTOR_PATH):
                self.easyocr_detector_session = self._load_onnx_model(EASYOCR_DETECTOR_PATH, "EasyOCR Detector")
                if self.easyocr_detector_session:
                    print("[✅ EasyOCR] Detector 모델 로딩 완료!")
                else:
                    print("[❌ EasyOCR] Detector 모델 로딩 실패!")
                self._reset_console_color()
            else:
                print(f"[❌ EasyOCR] Detector 모델 파일 없음: {EASYOCR_DETECTOR_PATH}")
            
            if os.path.exists(EASYOCR_RECOGNIZER_PATH):
                self.easyocr_recognizer_session = self._load_onnx_model(EASYOCR_RECOGNIZER_PATH, "EasyOCR Recognizer")
                if self.easyocr_recognizer_session:
                    print("[✅ EasyOCR] Recognizer 모델 로딩 완료!")
                else:
                    print("[❌ EasyOCR] Recognizer 모델 로딩 실패!")
                self._reset_console_color()
            else:
                print(f"[❌ EasyOCR] Recognizer 모델 파일 없음: {EASYOCR_RECOGNIZER_PATH}")
        
        # YOLO ONNX 모델 (GPU 우선, CPU 폴백)
        if USE_YOLO_ONNX and os.path.exists(YOLO_ONNX_PATH):
            try:
                self.yolo_onnx_session = self._load_onnx_model(YOLO_ONNX_PATH, "YOLOv11 객체탐지")
            except Exception as e:
                print(f"[⚠️ YOLO] GPU 로딩 실패, CPU로 재시도: {e}")
                # CPU로만 로드
                self.yolo_onnx_session = ort.InferenceSession(
                    YOLO_ONNX_PATH,
                    providers=['CPUExecutionProvider']
                )
                print("[✅ YOLO] CPU 로딩 완료!")
            self._reset_console_color()  # YOLO 로딩 후에도 색상 리셋
        
        # NPU 세션 로딩 시도 (QNN 직접 사용)
        if USE_EASYOCR_ONNX:
            if os.path.exists(EASYOCR_DETECTOR_PATH) and os.path.exists(EASYOCR_RECOGNIZER_PATH):
                self._try_load_npu_sessions()
        
        # Nomic API 로그인 (폴백용)
        if NOMIC_API_AVAILABLE and not self.onnx_session:
            try:
                login(token=config.NOMIC_TOKEN)
                print("[✅ Nomic API 로그인 완료]")
            except:
                print("[⚠️ Nomic API 로그인 실패]")
    
    def _reset_console_color(self):
        """콘솔 색상 리셋 (ONNX 에러 후 색상 복구)"""
        import sys
        if sys.platform == "win32":
            import os
            os.system('')  # Windows 콘솔 ANSI 활성화
        print('\033[0m', end='')  # ANSI 리셋 코드
    
    def _load_onnx_model(self, model_path, model_name):
        """ONNX 모델 로딩 (GPU 우선, CPU 폴백)"""
        # ONNX 로그 레벨을 WARNING으로 설정 (에러 메시지 숨김)
        ort.set_default_logger_severity(3)  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
        
        providers_to_try = [
            ('CUDAExecutionProvider', {'device_id': 0}),  # GPU
            ('CPUExecutionProvider', {})  # CPU
        ]
        
        for provider_name, provider_options in providers_to_try:
            try:
                device_type = "GPU" if "CUDA" in provider_name else "CPU"
                print(f"[🚀 ONNX] {model_name} 모델 로딩 시도 ({device_type})...")
                
                session = ort.InferenceSession(
                    model_path,
                    providers=[provider_name],
                    provider_options=[provider_options]
                )
                
                # 실제 사용 중인 프로바이더 확인
                actual_provider = session.get_providers()[0]
                actual_device = "GPU" if "CUDA" in actual_provider else "CPU"
                
                self._reset_console_color()
                print(f"[✅ ONNX] {model_name} 모델 로딩 완료! ({actual_device}: {actual_provider})")
                
                return session
                
            except Exception as e:
                # GPU 실패는 조용히 처리, CPU 실패만 로그
                if "CUDA" not in provider_name:
                    print(f"[❌ ONNX] {model_name} {device_type} 로딩 실패: {e}")
                continue
        
        print(f"[❌ ONNX] {model_name} 모델 로딩 실패 - 모든 프로바이더 시도함")
        return None
    
    def _try_load_npu_sessions(self):
        """NPU 세션 로딩 시도 (메인 프로세스에서 직접)"""
        try:
            print("[🚀 NPU] 메인 프로세스에서 NPU 세션 로딩 시도...")
            
            # QNN 세션 옵션
            so = ort.SessionOptions()
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
            
            ep_opts = {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "high_performance",
            }
            
            # Detector 세션 생성
            try:
                print("[🚀 NPU] Detector 세션 생성 중...")
                self.npu_detector_session = ort.InferenceSession(
                    EASYOCR_DETECTOR_PATH,
                    sess_options=so,
                    providers=["QNNExecutionProvider"],
                    provider_options=[ep_opts]
                )
                print("[✅ NPU] Detector 세션 생성 완료!")
            except Exception as e:
                print(f"[❌ NPU] Detector 세션 실패: {e}")
                self.npu_detector_session = None
            
            # NPU Recognizer 세션 생성
            try:
                print("[🚀 NPU] Recognizer 세션 생성 중...")
                self.npu_recognizer_session = ort.InferenceSession(
                    EASYOCR_RECOGNIZER_PATH,
                    sess_options=so,
                    providers=["QNNExecutionProvider"],
                    provider_options=[ep_opts]
                )
                print("[✅ NPU] Recognizer 세션 생성 완료!")
            except Exception as e:
                print(f"[❌ NPU] Recognizer 세션 실패: {e}")
                self.npu_recognizer_session = None
            
            # YOLO NPU 세션 생성 (선택사항)
            if os.path.exists(YOLO_ONNX_PATH):
                try:
                    print("[🚀 NPU] YOLO 세션 생성 중...")
                    self.npu_yolo_session = ort.InferenceSession(
                        YOLO_ONNX_PATH,
                        sess_options=so,
                        providers=["QNNExecutionProvider"],
                        provider_options=[ep_opts]
                    )
                    print("[✅ NPU] YOLO 세션 생성 완료!")
                except Exception as e:
                    print(f"[❌ NPU] YOLO 세션 실패: {e}")
                    self.npu_yolo_session = None
            
            if self.npu_detector_session and self.npu_recognizer_session:
                print("[🎉 NPU] 완전한 NPU 파이프라인 세션 준비 완료!")
            else:
                print("[⚠️ NPU] 일부 NPU 세션 로딩 실패")
                
        except Exception as e:
            print(f"[❌ NPU] 메인 프로세스 NPU 로딩 실패: {e}")
            # NPU 세션 모두 None으로 설정
            self.npu_detector_session = None
            self.npu_recognizer_session = None
            self.npu_yolo_session = None
    
    def load_yolo_model(self):
        """YOLO 모델 로딩"""
        if self.yolo_model is None:
            try:
                print("[🤖 YOLOv8 모델 로딩 시작]")
                self.yolo_model = YOLO(self.config.YOLO_MODEL)
                print("[✅ YOLOv8 모델 로딩 완료]")
                return True
            except Exception as e:
                print(f"[❗YOLO 모델 로딩 실패] {str(e)}")
                return False
        return True
    
    def load_qwen_model(self):
        """Qwen 모델 로딩 (GPU 우선, CPU 폴백)"""
        if self.qwen_model is None:
            print("[🤖 Qwen 모델 로딩 시작]")
            
            # 토크나이저 로딩
            try:
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.QWEN_MODEL, 
                    trust_remote_code=True
                )
                print("[✅ Qwen 토크나이저 로딩 완료]")
            except Exception as e:
                print(f"[❌ Qwen 토크나이저 로딩 실패] {str(e)}")
                return False
            
            # 모델 로딩 (GPU 우선 시도)
            if torch.cuda.is_available():
                try:
                    print("[🚀 Qwen] GPU 로딩 시도...")
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(
                        self.config.QWEN_MODEL,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        device_map=None
                    )
                    self.qwen_model.to("cuda")
                    self.qwen_model.eval()
                    print("[✅ Qwen] GPU 로딩 완료!")
                    return True
                except Exception as e:
                    print(f"[⚠️ Qwen] GPU 로딩 실패, CPU로 폴백: {str(e)}")
            
            # CPU 폴백
            try:
                print("[🚀 Qwen] CPU 로딩 시도...")
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    self.config.QWEN_MODEL,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.qwen_model.to("cpu")
                self.qwen_model.eval()
                print("[✅ Qwen] CPU 로딩 완료!")
                return True
            except Exception as e:
                print(f"[❌ Qwen] CPU 로딩도 실패: {str(e)}")
                return False
        return True
    
    def load_ocr_model(self):
        """OCR 모델 로딩 (ONNX 우선, EasyOCR API 폴백)"""
        # ONNX 모델이 이미 로딩되어 있으면
        if self.easyocr_detector_session and self.easyocr_recognizer_session:
            return True
            
        # EasyOCR API 폴백
        if self.ocr_reader is None:
            try:
                print("[📖 EasyOCR API 모델 로딩 시작 (폴백)]")
                self.ocr_reader = easyocr.Reader(['ko', 'en'])
                print("[✅ EasyOCR API 모델 로딩 완료]")
                return True
            except Exception as e:
                print(f"[❗EasyOCR API 모델 로딩 실패] {str(e)}")
                return False
        return True
    
    def _preprocess_image_for_onnx(self, image_np):
        """ONNX OCR을 위한 이미지 전처리"""
        import cv2
        
        # 이미지 크기 조정 (ONNX 모델 요구사항: 608x800)
        target_height, target_width = 608, 800
        
        # 이미지를 정확한 크기로 리사이즈
        resized = cv2.resize(image_np, (target_width, target_height))
        print(f"[🔧 ONNX OCR] 이미지 리사이즈: {image_np.shape[:2]} → {resized.shape[:2]}")
        
        # 정규화 (0-255 -> 0-1)
        normalized = resized.astype(np.float32) / 255.0
        
        # CHW 형식으로 변환 (Height, Width, Channel -> Channel, Height, Width)
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # 배치 차원 추가 (1, C, H, W)
        batched = np.expand_dims(transposed, axis=0)
        
        print(f"[🔧 ONNX OCR] 최종 입력 형태: {batched.shape}")
        return batched
    
    def _postprocess_ocr_result(self, onnx_output, original_image_shape):
        """ONNX OCR 결과 후처리"""
        detections = []
        
        try:
            # ONNX 출력: results [1, 304, 400, 2], features [1, 32, 304, 400]
            results = onnx_output[0]  # [1, 304, 400, 2]
            features = onnx_output[1]  # [1, 32, 304, 400]
            
            print(f"[🔧 ONNX OCR] 결과 형태: {results.shape}, 특징 형태: {features.shape}")
            
            # 텍스트 영역 탐지 (간단한 임계값 기반)
            # results의 마지막 차원이 [score, class] 또는 [x, y] 좌표일 가능성
            batch_results = results[0]  # [304, 400, 2]
            
            # 임계값 이상의 영역 찾기
            threshold = 0.5
            text_regions = []
            
            for i in range(batch_results.shape[0]):
                for j in range(batch_results.shape[1]):
                    score = batch_results[i, j, 0]  # 첫 번째 값을 신뢰도로 가정
                    if score > threshold:
                        # 좌표 계산 (원본 이미지 크기로 스케일링)
                        orig_h, orig_w = original_image_shape[:2]
                        x = j * orig_w / 400  # 400은 모델 출력 너비
                        y = i * orig_h / 304  # 304는 모델 출력 높이
                        
                        # EasyOCR 형식으로 변환: [좌표, 텍스트, 신뢰도]
                        detection = [
                            [[x, y], [x+20, y], [x+20, y+10], [x, y+10]],  # 바운딩 박스
                            f"Text_{len(text_regions)}",  # 더미 텍스트 (실제 OCR 필요)
                            float(score)
                        ]
                        text_regions.append(detection)
            
            print(f"[✅ ONNX OCR] {len(text_regions)}개 텍스트 영역 탐지됨")
            return text_regions[:10]  # 최대 10개만 반환
            
        except Exception as e:
            print(f"[❌ ONNX OCR] 후처리 실패: {e}")
            return []
    
    def extract_text_from_image_onnx(self, image_np):
        """EasyOCR 방식으로 이미지에서 텍스트 추출 (NPU 우선, 낮은 임계값)"""
        print(f"[🔍 DEBUG] Detector 세션: {self.easyocr_detector_session is not None}")
        print(f"[🔍 DEBUG] Recognizer 세션: {self.easyocr_recognizer_session is not None}")
        
        # 1. 메인 프로세스 완전한 NPU 파이프라인 우선 시도
        if self.npu_detector_session and self.npu_recognizer_session:
            try:
                print("[🚀 NPU Direct] 완전한 NPU 파이프라인 실행...")
                result = self._process_with_npu_direct(image_np)
                if result is not None:  # 빈 리스트도 성공으로 처리
                    print("[✅ NPU Direct] 성공")
                    return result
            except Exception as e:
                print(f"[⚠️ NPU Direct] 실패, 서브프로세스 시도: {e}")
        
        # 2. 서브프로세스 NPU 시도 (폴백)
        try:
            print("[🚀 NPU Subprocess] 서브프로세스 NPU 실행...")
            result = self._run_npu_easyocr_via_subprocess(image_np)
            if result is not None:  # 빈 리스트도 성공으로 처리
                print("[✅ NPU Subprocess] 성공")
                return result
        except Exception as e:
            print(f"[⚠️ NPU Subprocess] 실패, ONNX로 폴백: {e}")
        
        # 2. ONNX 폴백
        if not self.easyocr_detector_session or not self.easyocr_recognizer_session:
            print("[❌ OCR DEBUG] ONNX 모델 세션이 None입니다")
            return None
        
        print("[🚀 EasyOCR ONNX] 폴백 처리 시작...")
        try:
            return self._process_with_simple_pipeline(image_np)
        except Exception as e:
            print(f"[❌ EasyOCR ONNX] 실패, 기존 방식으로 폴백: {e}")
            return self._process_with_manual_method(image_np)
    
    def load_summarizer(self):
        """요약 모델 로딩 (비활성화 - Qwen 사용)"""
        print("[ℹ️ BART 요약 모델 비활성화됨 - Qwen 사용]")
        return False  # 항상 False 반환하여 Qwen 사용 강제
    
    # ---------------- NPU 선시도: 외부 QNN venv에서 임베딩 생성 ----------------
    def _run_npu_embed_via_subprocess(self, texts):
        """
        QNN 전용 venv를 PowerShell에서 활성화한 뒤,
        별도 임시 파이썬 스크립트를 실행해 NPU(HTP)로 임베딩을 생성.
        성공하면 float32 ndarray 리스트를 반환, 실패하면 None.
        """
        try:
            # 임시 파일 준비
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                in_json = td_path / "input.json"
                out_npy = td_path / "embeddings.npy"
                script_py = td_path / "npu_embed_runner.py"

                # 입력 데이터 저장
                payload = {
                    "texts": texts,
                    "onnx_model_path": QNN_ONNX_MODEL_PATH
                }
                in_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

                # 임시 파이썬 스크립트 작성 (네가 준 NPU 코드 + I/O)
                script_code = textwrap.dedent(rf"""
                    import json, sys, numpy as np, onnxruntime as ort, time
                    from transformers import AutoTokenizer

                    in_json = r"{in_json}"
                    out_npy = r"{out_npy}"

                    data = json.loads(open(in_json, "r", encoding="utf-8").read())
                    texts = data["texts"]
                    onnx_model_path = data["onnx_model_path"]

                    so = ort.SessionOptions()
                    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

                    ep_opts = {{
                        "backend_path": "QnnHtp.dll",
                        "htp_performance_mode": "high_performance",
                    }}

                    session = ort.InferenceSession(
                        onnx_model_path,
                        sess_options=so,
                        providers=["QNNExecutionProvider"],
                        provider_options=[ep_opts],
                    )

                    in_names = [i.name for i in session.get_inputs()]
                    out_name = session.get_outputs()[0].name

                    def pick(cands, target):
                        for nm in cands:
                            low = nm.lower()
                            if target=="tokens" and "token" in low: return nm
                            if target=="mask" and "mask" in low: return nm
                        return None

                    name_tokens = "input_tokens" if "input_tokens" in in_names else pick(in_names, "tokens")
                    name_mask   = "attention_masks" if "attention_masks" in in_names else pick(in_names, "mask")
                    assert name_tokens and name_mask, f"입력 이름 매핑 실패: {{in_names}}"

                    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", use_fast=True)
                    MAX_LEN = 128

                    def enc_one(t):
                        e = tokenizer(t, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="np")
                        return {{
                            name_tokens: e["input_ids"].astype(np.int32),
                            name_mask:   e["attention_mask"].astype(np.float32),
                        }}

                    # 워밍업
                    if len(texts) > 0:
                        wf = enc_one(texts[0])
                        for _ in range(2): _ = session.run([out_name], wf)

                    embs = []
                    for t in texts:
                        out = session.run([out_name], enc_one(t))[0]  # (1, D)
                        embs.append(out[0].astype(np.float32))        # raw embedding (정규화 없음)

                    arr = np.stack(embs, axis=0)  # (N, D)
                    np.save(out_npy, arr)
                """).strip()
                script_py.write_text(script_code, encoding="utf-8")

                # PowerShell 명령 구성: QNN venv 활성화 → 워킹 디렉토리 이동 → 파이썬 실행
                ps_cmd = (
                    f"& {{"
                    f"  cd '{QNN_DIR_PATH}'; "
                    f"  . '{QNN_SETUP_PS1}'; "
                    f"  Activate_ORT_QNN_VENV -rootDirPath '{QNN_DIR_PATH}'; "
                    f"  Set-Location '{os.getcwd()}'; "  # 백엔드 워킹디렉토리 유지
                    f"  $env:PYTHONUTF8='1'; "
                    f"  python '{script_py}'; "
                    f"}}"
                )

                # 실행
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300
                )

                # 로그 출력(참고)
                if result.stdout:
                    print("[ℹ️ NPU stdout]", result.stdout.strip())
                if result.stderr:
                    print("[ℹ️ NPU stderr]", result.stderr.strip())

                # 결과 확인
                if result.returncode == 0 and out_npy.exists():
                    embs = np.load(out_npy).astype(np.float32)
                    # list[np.ndarray]로 변환 (기존 반환 형식과 호환)
                    return [embs[i] for i in range(embs.shape[0])]
                else:
                    print(f"[⚠️ NPU] 서브프로세스 실패(returncode={result.returncode})")
                    return None

        except Exception as e:
            print(f"[⚠️ NPU] 실행 실패: {e}")
            return None

    def _get_embeddings(self, texts):
        """텍스트 임베딩 생성 (ONNX 우선, API 폴백)
        - 먼저 NPU(QNN) 외부 실행을 시도
        - 실패하면 기존 경로(현재 venv의 ONNX 세션 → Nomic API)로 폴백
        """
        # 1) NPU(QNN) 외부 실행 선시도
        try:
            print("[🚀 NPU] 외부 QNN venv에서 임베딩 시도...")
            npu_embs = self._run_npu_embed_via_subprocess(texts)
            if npu_embs and len(npu_embs) == len(texts):
                print("[✅ NPU] 임베딩 생성 성공 (QNN/HTP)")
                return {'embeddings': npu_embs}
            else:
                print("[ℹ️ NPU] 임베딩 미생성 또는 개수 불일치 → 폴백 진행")
        except Exception as e:
            print(f"[⚠️ NPU] 예외 발생 → 폴백: {e}")

        # 2) 현재 venv의 ONNX 세션 사용 (GPU → CPU)
        if self.onnx_session and self.bert_tokenizer:
            # ONNX 모델 사용
            try:
                print(f"[🚀 ONNX] 임베딩 생성 시작 - {len(texts)}개 텍스트")
                embeddings = []
                for i, text in enumerate(texts):
                    inputs = self.bert_tokenizer(
                        text, 
                        padding="max_length", 
                        max_length=128, 
                        truncation=True,
                        return_tensors="np"
                    )
                    
                    outputs = self.onnx_session.run(None, {
                        "input_tokens": inputs["input_ids"].astype(np.int32),
                        "attention_masks": inputs["attention_mask"].astype(np.float32)
                    })
                    embeddings.append(outputs[0][0])
                    print(f"[✅ ONNX] 텍스트 {i+1}/{len(texts)} 임베딩 완료 (차원: {len(outputs[0][0])})")
                
                print(f"[🎉 ONNX] 전체 임베딩 생성 완료!")
                return {'embeddings': embeddings}
            except Exception as e:
                print(f"[⚠️ ONNX] 임베딩 생성 실패: {e}")
        
        # 3) Nomic API 사용 (폴백)
        if NOMIC_API_AVAILABLE:
            return embed.text(texts, model='nomic-embed-text-v1', task_type='classification')
        else:
            raise Exception("임베딩 모델을 사용할 수 없습니다.")
    
    def classify_email(self, text):
        """이메일 분류"""
        try:
            text_inputs = [text] + self.config.CANDIDATE_LABELS
            result = self._get_embeddings(text_inputs)
            
            embedding_list = result['embeddings']
            email_embedding = [embedding_list[0]]
            label_embeddings = embedding_list[1:]
            
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity(email_embedding, label_embeddings)[0]
            best_index = scores.argmax()
            
            return {
                'classification': self.config.CANDIDATE_LABELS[best_index],
                'confidence': scores[best_index]
            }
            
        except Exception as e:
            print(f"[⚠️ 분류 실패] {str(e)}")
            return {'classification': 'unknown', 'confidence': 0.0}
    
    def _run_npu_easyocr_via_subprocess(self, image_np):
        """
        QNN venv를 PowerShell에서 활성화한 뒤,
        별도 임시 파이썬 스크립트를 실행해 NPU(HTP)로 EasyOCR을 실행.
        성공하면 텍스트 영역 리스트를 반환, 실패하면 None.
        """
        try:
            import tempfile, json, os
            from pathlib import Path
            
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                in_json = td_path / "easyocr_input.json"
                out_json = td_path / "easyocr_output.json"
                script_py = td_path / "npu_easyocr_runner.py"
                img_path = td_path / "input_image.npy"
                
                # 이미지 저장
                np.save(img_path, image_np)
                
                # EasyOCR Reader 문자셋 가져오기
                if not hasattr(self, '_easyocr_reader'):
                    import easyocr
                    self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                    print(f"[🔍 EasyOCR] Reader 초기화 완료 - 문자셋: {len(self._easyocr_reader.character)}개")
                
                # 입력 데이터 저장 (문자셋 포함)
                payload = {
                    "image_path": str(img_path),
                    "detector_path": EASYOCR_DETECTOR_PATH,
                    "recognizer_path": EASYOCR_RECOGNIZER_PATH,
                    "charset": list(self._easyocr_reader.character)  # 실제 EasyOCR 문자셋 전달
                }
                in_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                
                # NPU EasyOCR 스크립트 작성
                script_code = textwrap.dedent(rf"""
                    import json, sys, numpy as np, onnxruntime as ort, cv2
                    from pathlib import Path
                    
                    in_json = r"{in_json}"
                    out_json = r"{out_json}"
                    
                    try:
                        # 입력 로드
                        data = json.loads(open(in_json, "r", encoding="utf-8").read())
                        image_np = np.load(data["image_path"])
                        detector_path = data["detector_path"]
                        recognizer_path = data["recognizer_path"]
                        
                        # QNN 세션 옵션
                        so = ort.SessionOptions()
                        so.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
                        
                        ep_opts = {{
                            "backend_path": "QnnHtp.dll",
                            "htp_performance_mode": "high_performance",
                        }}
                        
                        # Detector 세션
                        detector_session = ort.InferenceSession(
                            detector_path,
                            sess_options=so,
                            providers=["QNNExecutionProvider"],
                            provider_options=[ep_opts]
                        )
                        
                        # Recognizer 세션
                        recognizer_session = ort.InferenceSession(
                            recognizer_path,
                            sess_options=so,
                            providers=["QNNExecutionProvider"],
                            provider_options=[ep_opts]
                        )
                        
                        print("[NPU EasyOCR] 모델 로딩 완료")
                        
                        # EasyOCR 전처리 (기존 로직 복사)
                        from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
                        from easyocr.imgproc import normalizeMeanVariance
                        
                        target_height, target_width = 608, 800
                        resized = cv2.resize(image_np, (target_width, target_height))
                        normalized = normalizeMeanVariance(resized)
                        transposed = np.transpose(normalized, (2, 0, 1))
                        batched = np.expand_dims(transposed, axis=0).astype(np.float32)
                        
                        # Detector 실행
                        detector_output = detector_session.run(None, {{"image": batched}})
                        results = detector_output[0]
                        
                        # Detector 후처리
                        score_text = results[0][:, :, 0]
                        score_link = results[0][:, :, 1]
                        
                        print(f"[🔍 NPU DEBUG] 스코어 범위: text={{score_text.min():.4f}}~{{score_text.max():.4f}}, link={{score_link.min():.4f}}~{{score_link.max():.4f}}")
                        print(f"[🔍 NPU DEBUG] 스코어 형태: text={{score_text.shape}}, link={{score_link.shape}}")
                        
                        # 0.01 이상인 픽셀 개수 확인
                        high_text_count = np.sum(score_text > 0.01)
                        high_link_count = np.sum(score_link > 0.01) 
                        print(f"[🔍 NPU DEBUG] 0.01 이상 픽셀: text={{high_text_count}}, link={{high_link_count}}")
                        
                        boxes, polys, mapper = getDetBoxes(
                            score_text, score_link,
                            text_threshold=0.2, link_threshold=0.15, low_text=0.15, poly=False
                        )
                        
                        print(f"[🔍 NPU DEBUG] getDetBoxes 결과: {{len(boxes)}}개 박스")
                        if len(boxes) > 0:
                            print(f"[🔍 NPU DEBUG] 첫 번째 박스: {{boxes[0] if boxes[0] is not None else 'None'}}")
                        
                        if len(boxes) == 0:
                            with open(out_json, "w", encoding="utf-8") as f:
                                json.dump({{"success": True, "text_regions": []}}, f)
                            sys.exit(0)
                        
                        # 좌표 조정 (백업 파일 방식)
                        orig_h, orig_w = image_np.shape[:2]
                        ratio_w = orig_w / target_width
                        ratio_h = orig_h / target_height
                        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
                        
                        # 각 박스에서 텍스트 인식 (실제 Recognizer 사용)
                        text_regions = []
                        for i, box in enumerate(boxes):  # 모든 박스 처리
                            if box is None or len(box) != 4:
                                continue
                                
                            # 박스 좌표 추출
                            xs = [p[0] for p in box]
                            ys = [p[1] for p in box]
                            x1, x2 = int(min(xs)), int(max(xs))
                            y1, y2 = int(min(ys)), int(max(ys))
                            
                            if (x2 - x1) < 5 or (y2 - y1) < 5:
                                continue
                            
                            # 이미지 크롭
                            cropped = image_np[y1:y2, x1:x2]
                            if cropped.size == 0:
                                continue
                            
                            # Recognizer로 실제 텍스트 인식
                            try:
                                # 그레이스케일 변환
                                if len(cropped.shape) == 3:
                                    gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                                else:
                                    gray_image = cropped
                                
                                # PIL Image로 변환
                                from PIL import Image
                                pil_image = Image.fromarray(gray_image).convert('L')
                                
                                # Recognizer 입력 크기 (64x1000)
                                imgH, imgW = 64, 1000
                                
                                # 비율 유지하며 리사이즈
                                h, w = pil_image.size[1], pil_image.size[0]  # PIL은 (w, h)
                                ratio = min(imgW / w, imgH / h)
                                new_w = int(w * ratio)
                                new_h = int(h * ratio)
                                
                                resized = pil_image.resize((new_w, new_h), Image.LANCZOS)
                                
                                # 패딩 추가 (중앙 정렬)
                                padded = Image.new('L', (imgW, imgH), color=0)
                                paste_x = (imgW - new_w) // 2
                                paste_y = (imgH - new_h) // 2
                                padded.paste(resized, (paste_x, paste_y))
                                
                                # NumPy 배열로 변환 및 정규화
                                img_array = np.array(padded).astype(np.float32) / 255.0
                                
                                # 배치 및 채널 차원 추가 (1, 1, 64, 1000)
                                input_tensor = np.expand_dims(np.expand_dims(img_array, 0), 0)
                                
                                # Recognizer 실행
                                recognizer_outputs = recognizer_session.run(None, {{"image": input_tensor}})
                                
                                # 원본 ai_models.py와 동일한 방식으로 디코딩
                                try:
                                    # EasyOCR Reader 초기화 (converter 가져오기 위해)
                                    import easyocr
                                    import torch
                                    import torch.nn.functional as F
                                    
                                    temp_reader = easyocr.Reader(['en'], gpu=False)
                                    
                                    # 출력을 PyTorch Tensor로 변환
                                    output = recognizer_outputs[0]
                                    preds = torch.from_numpy(output).float()
                                    
                                    print(f"[🔍 NPU CTC] 입력 형태: {{preds.shape}}")
                                    
                                    # Softmax 적용 (원본과 동일)
                                    preds = F.softmax(preds, dim=2)
                                    
                                    # Greedy 디코딩 (원본과 동일)
                                    _, preds_index = preds.max(2)
                                    
                                    # 길이 정보 생성
                                    batch_size = preds.size(0)
                                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                                    
                                    # 실제 EasyOCR Converter로 디코딩 (원본과 동일)
                                    converter = temp_reader.converter
                                    preds_str = converter.decode_greedy(
                                        preds_index.view(-1).data.cpu().detach().numpy(), 
                                        preds_size.data
                                    )
                                    
                                    recognized_text = preds_str[0] if preds_str else ""
                                    print(f"[✅ NPU CTC] 원본 방식 디코딩: '{{recognized_text}}'")
                                    
                                except Exception as converter_e:
                                    print(f"[⚠️ NPU CTC] EasyOCR converter 실패, 간소화 방식 사용: {{converter_e}}")
                                    # 폴백: 간소화된 CTC 디코딩
                                    output_probs = recognizer_outputs[0][0]
                                    predicted_ids = np.argmax(output_probs, axis=1)
                                    charset = data.get("charset", [])
                                    if len(charset) > 1:
                                        decoded_chars = []
                                        prev_char = -1
                                        for char_id in predicted_ids:
                                            if char_id != prev_char and char_id > 0 and char_id < len(charset):
                                                decoded_chars.append(charset[char_id])
                                            prev_char = char_id
                                        recognized_text = ''.join(decoded_chars).strip()
                                    else:
                                        recognized_text = ""
                                
                                if recognized_text:
                                    text_regions.append([
                                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                        recognized_text,
                                        0.8
                                    ])
                                    print(f"[✅ NPU OCR] Box {{i+1}}: '{{recognized_text}}'")
                                else:
                                    # 빈 텍스트인 경우 더미로 대체
                                    text_regions.append([
                                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                        f"NPU_Text_{{i+1}}",
                                        0.5
                                    ])
                                    
                            except Exception as recog_e:
                                print(f"[⚠️ NPU OCR] Recognizer 실패 (Box {{i+1}}): {{recog_e}}")
                                # 실패 시 더미 텍스트
                                text_regions.append([
                                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                    f"NPU_Text_{{i+1}}",
                                    0.3
                                ])
                        
                        # 결과 저장
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump({{"success": True, "text_regions": text_regions}}, f)
                        
                    except Exception as e:
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump({{"success": False, "error": str(e)}}, f)
                """).strip()
                
                script_py.write_text(script_code, encoding="utf-8")
                
                # PowerShell 명령 (Nomic과 동일한 구조)
                ps_cmd = (
                    f"& {{"
                    f"  cd '{QNN_DIR_PATH}'; "
                    f"  . '{QNN_SETUP_PS1}'; "
                    f"  Activate_ORT_QNN_VENV -rootDirPath '{QNN_DIR_PATH}'; "
                    f"  Set-Location '{os.getcwd()}'; "
                    f"  $env:PYTHONUTF8='1'; "
                    f"  python '{script_py}'; "
                    f"}}"
                )
                
                # 실행
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300
                )
                
                # 결과 확인
                if result.returncode == 0 and out_json.exists():
                    output_data = json.loads(out_json.read_text(encoding="utf-8"))
                    if output_data.get("success"):
                        print(f"[✅ NPU EasyOCR] {len(output_data['text_regions'])}개 영역 탐지됨")
                        return output_data["text_regions"]
                    else:
                        print(f"[❌ NPU EasyOCR] 스크립트 오류: {output_data.get('error')}")
                else:
                    print(f"[⚠️ NPU EasyOCR] 프로세스 실패(returncode={result.returncode})")
                    if result.stderr:
                        print(f"[stderr] {result.stderr}")
                
                return None
                
        except Exception as e:
            print(f"[⚠️ NPU EasyOCR] 실행 실패: {e}")
            return None
    
    def _run_npu_yolo_via_subprocess(self, image_np):
        """
        QNN venv를 PowerShell에서 활성화한 뒤,
        별도 임시 파이썬 스크립트를 실행해 NPU(HTP)로 YOLO를 실행.
        성공하면 탐지 결과 리스트를 반환, 실패하면 None.
        """
        try:
            import tempfile, json, os
            from pathlib import Path
            
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                in_json = td_path / "yolo_input.json"
                out_json = td_path / "yolo_output.json"
                script_py = td_path / "npu_yolo_runner.py"
                img_path = td_path / "input_image.npy"
                
                # 이미지 저장
                np.save(img_path, image_np)
                
                # 입력 데이터 저장
                payload = {
                    "image_path": str(img_path),
                    "yolo_path": YOLO_ONNX_PATH
                }
                in_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                
                # NPU YOLO 스크립트 작성
                script_code = textwrap.dedent(rf"""
                    import json, sys, numpy as np, onnxruntime as ort, cv2
                    from pathlib import Path
                    
                    in_json = r"{in_json}"
                    out_json = r"{out_json}"
                    
                    try:
                        # 입력 로드
                        data = json.loads(open(in_json, "r", encoding="utf-8").read())
                        image_np = np.load(data["image_path"])
                        yolo_path = data["yolo_path"]
                        
                        # QNN 세션 옵션
                        so = ort.SessionOptions()
                        so.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
                        
                        ep_opts = {{
                            "backend_path": "QnnHtp.dll",
                            "htp_performance_mode": "high_performance",
                        }}
                        
                        # YOLO 세션
                        yolo_session = ort.InferenceSession(
                            yolo_path,
                            sess_options=so,
                            providers=["QNNExecutionProvider"],
                            provider_options=[ep_opts]
                        )
                        
                        print("[NPU YOLO] 모델 로딩 완료")
                        
                        # YOLO 전처리 (기존 로직 복사)
                        input_size = 640
                        original_height, original_width = image_np.shape[:2]
                        
                        # 비율 유지하면서 리사이즈
                        scale = min(input_size / original_width, input_size / original_height)
                        new_width = int(original_width * scale)
                        new_height = int(original_height * scale)
                        
                        # 리사이즈
                        resized = cv2.resize(image_np, (new_width, new_height))
                        
                        # 패딩 추가 (중앙 정렬)
                        pad_x = (input_size - new_width) // 2
                        pad_y = (input_size - new_height) // 2
                        
                        padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
                        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
                        
                        # 차원 변경 (HWC -> CHW) - uint8 유지
                        transposed = np.transpose(padded, (2, 0, 1))
                        batched = np.expand_dims(transposed, axis=0)
                        
                        # YOLO 실행
                        input_name = yolo_session.get_inputs()[0].name
                        outputs = yolo_session.run(None, {{input_name: batched}})
                        
                        # YOLO 후처리 (ONNX와 완전히 동일하게)
                        predictions = outputs[0][0]  # (1, 84, 8400) -> (84, 8400)
                        
                        # 좌표와 신뢰도 분리
                        boxes = predictions[:4]  # x, y, w, h
                        scores = predictions[4:]  # 클래스별 신뢰도 (80, 8400)
                        
                        # 최대 신뢰도와 클래스 ID
                        class_scores = np.max(scores, axis=0)  # (8400,)
                        class_ids = np.argmax(scores, axis=0)  # (8400,)
                        
                        # 신뢰도 임계값 적용
                        conf_threshold = 0.5
                        valid_indices = class_scores > conf_threshold
                        
                        if not np.any(valid_indices):
                            with open(out_json, "w", encoding="utf-8") as f:
                                json.dump({{"success": True, "detections": []}}, f)
                            sys.exit(0)
                        
                        valid_boxes = boxes[:, valid_indices]  # (4, N)
                        valid_scores = class_scores[valid_indices]  # (N,)
                        valid_class_ids = class_ids[valid_indices]  # (N,)
                        
                        # 좌표 변환 (중심점, 너비, 높이 -> x1, y1, x2, y2)
                        x_center = valid_boxes[0]
                        y_center = valid_boxes[1]
                        width = valid_boxes[2]
                        height = valid_boxes[3]
                        
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        
                        # 원본 이미지 좌표로 역변환
                        x1 = (x1 - pad_x) / scale
                        y1 = (y1 - pad_y) / scale
                        x2 = (x2 - pad_x) / scale
                        y2 = (y2 - pad_y) / scale
                        
                        # 좌표 클리핑
                        x1 = np.clip(x1, 0, original_width)
                        y1 = np.clip(y1, 0, original_height)
                        x2 = np.clip(x2, 0, original_width)
                        y2 = np.clip(y2, 0, original_height)
                        
                        # 간단한 NMS (OpenCV 없이)
                        keep_indices = []
                        order = valid_scores.argsort()[::-1]
                        
                        while len(order) > 0:
                            i = order[0]
                            keep_indices.append(i)
                            
                            if len(order) == 1:
                                break
                            
                            # IoU 계산
                            box1 = [x1[i], y1[i], x2[i], y2[i]]
                            remaining_boxes = np.column_stack([x1[order[1:]], y1[order[1:]], x2[order[1:]], y2[order[1:]]])
                            
                            # 간단한 IoU
                            x1_max = np.maximum(box1[0], remaining_boxes[:, 0])
                            y1_max = np.maximum(box1[1], remaining_boxes[:, 1])
                            x2_min = np.minimum(box1[2], remaining_boxes[:, 2])
                            y2_min = np.minimum(box1[3], remaining_boxes[:, 3])
                            
                            intersection_area = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)
                            
                            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            boxes_area = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
                            
                            union_area = box1_area + boxes_area - intersection_area
                            ious = intersection_area / (union_area + 1e-6)
                            
                            keep = np.where(ious <= 0.45)[0]
                            order = order[keep + 1]
                        
                        # COCO 클래스 이름
                        class_names = [
                            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        ]
                        
                        # 결과 구성
                        detections = []
                        for i in keep_indices[:10]:  # 최대 10개만 처리
                            class_id = int(valid_class_ids[i])
                            confidence = float(valid_scores[i])
                            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{{class_id}}"
                            
                            detections.append({{
                                'class': class_name,
                                'confidence': confidence,
                                'class_id': class_id,
                                'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])]
                            }})
                        
                        # 결과 저장
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump({{"success": True, "detections": detections}}, f)
                        
                    except Exception as e:
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump({{"success": False, "error": str(e)}}, f)
                """).strip()
                
                script_py.write_text(script_code, encoding="utf-8")
                
                # PowerShell 명령 (Nomic과 동일한 구조)
                ps_cmd = (
                    f"& {{"
                    f"  cd '{QNN_DIR_PATH}'; "
                    f"  . '{QNN_SETUP_PS1}'; "
                    f"  Activate_ORT_QNN_VENV -rootDirPath '{QNN_DIR_PATH}'; "
                    f"  Set-Location '{os.getcwd()}'; "
                    f"  $env:PYTHONUTF8='1'; "
                    f"  python '{script_py}'; "
                    f"}}"
                )
                
                # 실행
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300
                )
                
                # 결과 확인
                if result.returncode == 0 and out_json.exists():
                    output_data = json.loads(out_json.read_text(encoding="utf-8"))
                    if output_data.get("success"):
                        detections = output_data["detections"]
                        print(f"[✅ NPU YOLO] {len(detections)}개 객체 탐지됨")
                        # 개별 탐지 결과 출력
                        for detection in detections:
                            class_name = detection.get('class', 'unknown')
                            confidence = detection.get('confidence', 0.0)
                            print(f"[✅ NPU YOLO] 탐지: {class_name} ({confidence:.3f})")
                        return detections
                    else:
                        print(f"[❌ NPU YOLO] 스크립트 오류: {output_data.get('error')}")
                else:
                    print(f"[⚠️ NPU YOLO] 프로세스 실패(returncode={result.returncode})")
                    if result.stderr:
                        print(f"[stderr] {result.stderr}")
                
                return None
                
        except Exception as e:
            print(f"[⚠️ NPU YOLO] 실행 실패: {e}")
            return None
    
    def detect_objects_with_yolo_onnx(self, image_np):
        """YOLO ONNX 모델로 객체 탐지 (NPU 우선)"""
        # 1. NPU 우선 시도 (Nomic 방식)
        try:
            print("[🚀 NPU YOLO] NPU 실행 시도...")
            result = self._run_npu_yolo_via_subprocess(image_np)
            if result:
                print("[✅ NPU YOLO] 성공")
                return result
        except Exception as e:
            print(f"[⚠️ NPU YOLO] 실패, ONNX로 폴백: {e}")
        
        # 2. ONNX 세션 폴백
        if not self.yolo_onnx_session:
            print("[❌ YOLO ONNX] 세션이 None입니다")
            return []
            
        try:
            import cv2
            
            print("[🚀 YOLO ONNX] 객체 탐지 시작...")
            
            # YOLOv11 입력 크기 (640x640)
            input_size = 640
            
            # 이미지 전처리
            original_height, original_width = image_np.shape[:2]
            
            # 비율 유지하면서 리사이즈
            scale = min(input_size / original_width, input_size / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 리사이즈
            resized = cv2.resize(image_np, (new_width, new_height))
            
            # 패딩 추가 (중앙 정렬)
            pad_x = (input_size - new_width) // 2
            pad_y = (input_size - new_height) // 2
            
            padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)  # Gray padding
            padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
            
            # 차원 변경 (HWC -> CHW) - uint8 유지
            transposed = np.transpose(padded, (2, 0, 1))
            batched = np.expand_dims(transposed, axis=0)
            
            print(f"[🔧 YOLO] 전처리 완료: {image_np.shape} -> {batched.shape}")
            
            # ONNX 추론
            input_name = self.yolo_onnx_session.get_inputs()[0].name
            outputs = self.yolo_onnx_session.run(None, {input_name: batched})
            
            # YOLOv11 출력 후처리
            predictions = outputs[0][0]  # (1, 84, 8400) -> (84, 8400)
            
            print(f"[🔧 YOLO] 출력 형태: {predictions.shape}")
            
            # 좌표와 신뢰도 분리
            boxes = predictions[:4]  # x, y, w, h
            scores = predictions[4:]  # 클래스별 신뢰도
            
            # 최대 신뢰도와 클래스 ID
            class_scores = np.max(scores, axis=0)
            class_ids = np.argmax(scores, axis=0)
            
            # 신뢰도 임계값 적용
            conf_threshold = 0.5
            valid_indices = class_scores > conf_threshold
            
            if not np.any(valid_indices):
                print("[❌ YOLO] 신뢰도 임계값을 넘는 객체 없음")
                return []
            
            valid_boxes = boxes[:, valid_indices]
            valid_scores = class_scores[valid_indices]
            valid_class_ids = class_ids[valid_indices]
            
            # 좌표 변환 (중심점, 너비, 높이 -> x1, y1, x2, y2)
            x_center = valid_boxes[0]
            y_center = valid_boxes[1]
            width = valid_boxes[2]
            height = valid_boxes[3]
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # 원본 이미지 좌표로 역변환
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            
            # 좌표 클리핑
            x1 = np.clip(x1, 0, original_width)
            y1 = np.clip(y1, 0, original_height)
            x2 = np.clip(x2, 0, original_width)
            y2 = np.clip(y2, 0, original_height)
            
            # NMS 적용
            boxes_for_nms = np.column_stack([x1, y1, x2, y2])
            keep_indices = self._apply_nms(boxes_for_nms, valid_scores, iou_threshold=0.45)
            
            # COCO 클래스 이름 정의 (YOLOv11 기본)
            class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            
            # 결과 구성
            detections = []
            for i in keep_indices:
                class_id = int(valid_class_ids[i])
                confidence = float(valid_scores[i])
                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'class_id': class_id,
                    'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])]
                })
                
                print(f"[✅ YOLO] 탐지: {class_name} ({confidence:.3f})")
            
            print(f"[✅ YOLO ONNX] {len(detections)}개 객체 탐지됨")
            return detections
            
        except Exception as e:
            print(f"[❌ YOLO ONNX] 추론 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _apply_nms(self, boxes, scores, iou_threshold=0.45):
        """Non-Maximum Suppression 적용"""
        try:
            import cv2
            
            # OpenCV NMS 사용
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                scores.tolist(), 
                score_threshold=0.1, 
                nms_threshold=iou_threshold
            )
            
            if len(indices) > 0:
                return indices.flatten()
            else:
                return []
                
        except Exception as e:
            print(f"[⚠️ NMS] OpenCV NMS 실패, 수동 구현 사용: {e}")
            
            # 수동 NMS 구현
            indices = []
            order = scores.argsort()[::-1]  # 신뢰도 내림차순 정렬
            
            while len(order) > 0:
                i = order[0]
                indices.append(i)
                
                if len(order) == 1:
                    break
                
                # IoU 계산
                ious = self._calculate_iou(boxes[i], boxes[order[1:]])
                
                # IoU 임계값 이하인 박스들만 유지
                keep = np.where(ious <= iou_threshold)[0]
                order = order[keep + 1]
            
            return indices
    
    def _calculate_iou(self, box1, boxes):
        """IoU (Intersection over Union) 계산"""
        x1_max = np.maximum(box1[0], boxes[:, 0])
        y1_max = np.maximum(box1[1], boxes[:, 1])
        x2_min = np.minimum(box1[2], boxes[:, 2])
        y2_min = np.minimum(box1[3], boxes[:, 3])
        
        intersection_area = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union_area = box1_area + boxes_area - intersection_area
        
        return intersection_area / (union_area + 1e-6)
    
    def _process_with_npu_direct(self, image_np):
        """메인 프로세스에서 NPU 직접 실행 (서브프로세스 없음)"""
        import cv2
        from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
        from easyocr.imgproc import normalizeMeanVariance
        
        print("[🚀 NPU Direct] NPU 직접 처리 시작...")
        
        try:
            # 1. Detector 전처리
            target_height, target_width = 608, 800
            resized = cv2.resize(image_np, (target_width, target_height))
            normalized = normalizeMeanVariance(resized)
            transposed = np.transpose(normalized, (2, 0, 1))
            batched = np.expand_dims(transposed, axis=0).astype(np.float32)
            
            # 2. NPU Detector 실행
            detector_output = self.npu_detector_session.run(None, {"image": batched})
            results = detector_output[0]
            
            # 3. Detector 후처리
            score_text = results[0][:, :, 0]
            score_link = results[0][:, :, 1]
            
            print(f"[🔧 NPU] 스코어 범위: text={score_text.min():.4f}~{score_text.max():.4f}")
            
            boxes, polys, mapper = getDetBoxes(
                score_text, score_link,
                text_threshold=0.2, link_threshold=0.15, low_text=0.15, poly=False
            )
            
            print(f"[🔧 NPU] 탐지된 박스: {len(boxes)}개")
            
            if len(boxes) == 0:
                print("[❌ NPU] 텍스트 영역이 탐지되지 않음")
                return []
            
            # 좌표 조정
            orig_h, orig_w = image_np.shape[:2]
            ratio_w = orig_w / target_width
            ratio_h = orig_h / target_height
            boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
            
            # 4. 각 박스에서 텍스트 인식
            text_regions = []
            for i, box in enumerate(boxes):
                if box is None or len(box) != 4:
                    continue
                    
                # 박스 좌표 추출
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue
                
                # 이미지 크롭
                cropped = image_np[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                
                # NPU Recognizer로 텍스트 인식 (완전한 NPU 파이프라인)
                recognized_text = self._recognize_text_with_npu(cropped)
                
                if recognized_text and recognized_text != "???":
                    detection = [
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        recognized_text,
                        0.8
                    ]
                    text_regions.append(detection)
                    print(f"[✅ NPU] 완전한 NPU 파이프라인 Box {i+1}: '{recognized_text}'")
            
            print(f"[✅ NPU Direct] 완전한 NPU 파이프라인 - {len(text_regions)}개 텍스트 영역 인식됨")
            return text_regions
            
        except Exception as e:
            print(f"[❌ NPU Direct] 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def _recognize_text_with_npu(self, cropped_image):
        """NPU Recognizer로 텍스트 인식 (PyTorch converter 사용)"""
        try:
            import cv2
            from easyocr.recognition import AlignCollate
            from torch.utils.data import DataLoader, Dataset
            import torch
            
            # ListDataset 구현
            class SimpleListDataset(Dataset):
                def __init__(self, img_list):
                    self.img_list = img_list
                def __len__(self):
                    return len(self.img_list)
                def __getitem__(self, idx):
                    return self.img_list[idx]
            
            # 그레이스케일 변환
            if len(cropped_image.shape) == 3:
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = cropped_image
            
            # PIL Image로 변환
            from PIL import Image
            pil_image = Image.fromarray(gray_image).convert('L')
            
            # 전처리 (EasyOCR 방식)
            imgH, imgW = 64, 1000
            AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
            
            img_list = [pil_image]
            test_data = SimpleListDataset(img_list)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, 
                                   collate_fn=AlignCollate_normal, pin_memory=False)
            
            # 전처리된 이미지 가져오기
            for batch in test_loader:
                if isinstance(batch, torch.Tensor):
                    normalized = batch.numpy()
                else:
                    normalized = batch[0].numpy() if isinstance(batch[0], torch.Tensor) else batch[0]
                
                # NPU Recognizer 실행
                outputs = self.npu_recognizer_session.run(None, {"image": normalized})
                
                # PyTorch EasyOCR 방식으로 디코딩 (정확도 우선)
                print(f"[🔥 NPU] Recognizer 실행 완료")
                return self._decode_with_easyocr_converter(outputs[0])
            
            return ""
            
        except Exception as e:
            print(f"[⚠️ NPU Recognizer] 실패: {e}")
            return "???"
    
    def _recognize_text_with_onnx(self, cropped_image):
        """ONNX CPU Recognizer로 텍스트 인식 (하이브리드 방식)"""
        try:
            import cv2
            from easyocr.recognition import AlignCollate
            from torch.utils.data import DataLoader, Dataset
            import torch
            
            # ListDataset 구현
            class SimpleListDataset(Dataset):
                def __init__(self, img_list):
                    self.img_list = img_list
                def __len__(self):
                    return len(self.img_list)
                def __getitem__(self, idx):
                    return self.img_list[idx]
            
            # 그레이스케일 변환
            if len(cropped_image.shape) == 3:
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = cropped_image
            
            # PIL Image로 변환
            from PIL import Image
            pil_image = Image.fromarray(gray_image).convert('L')
            
            # 전처리 (EasyOCR 방식)
            imgH, imgW = 64, 1000
            AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
            
            img_list = [pil_image]
            test_data = SimpleListDataset(img_list)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, 
                                   collate_fn=AlignCollate_normal, pin_memory=False)
            
            # 전처리된 이미지 가져오기
            for batch in test_loader:
                if isinstance(batch, torch.Tensor):
                    normalized = batch.numpy()
                else:
                    normalized = batch[0].numpy() if isinstance(batch[0], torch.Tensor) else batch[0]
                
                # ONNX CPU Recognizer 실행
                outputs = self.easyocr_recognizer_session.run(None, {"image": normalized})
                
                # PyTorch EasyOCR 방식으로 디코딩 (정확도 우선)
                print(f"[🔥 ONNX CPU] Recognizer 실행 완료")
                return self._decode_with_easyocr_converter(outputs[0])
            
            return ""
            
        except Exception as e:
            print(f"[⚠️ ONNX Recognizer] 실패: {e}")
            return "???"
    
    
    def _process_with_simple_pipeline(self, image_np):
        """바탕화면 테스트에서 검증된 간단한 파이프라인"""
        import cv2
        from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
        from easyocr.imgproc import normalizeMeanVariance
        
        print("[🚀 Simple Pipeline] 간단한 OCR 파이프라인 시작...")
        
        try:
            # 1. Detector 전처리
            print("[🔧] Detector 전처리...")
            target_height, target_width = 608, 800
            resized = cv2.resize(image_np, (target_width, target_height))
            normalized = normalizeMeanVariance(resized)
            transposed = np.transpose(normalized, (2, 0, 1))
            batched = np.expand_dims(transposed, axis=0).astype(np.float32)
            
            # 2. Detector 실행
            print("[🔧] Detector 실행...")
            detector_output = self.easyocr_detector_session.run(None, {"image": batched})
            results = detector_output[0]  # [1, 304, 400, 2]
            
            # 3. Detector 후처리
            print("[🔧] Detector 후처리...")
            score_text = results[0][:, :, 0]
            score_link = results[0][:, :, 1]
            
            print(f"[🔧] 스코어 범위: text={score_text.min():.4f}~{score_text.max():.4f}")
            
            # 낮은 임계값 사용 (테스트에서 검증됨)
            boxes, polys, mapper = getDetBoxes(
                score_text, score_link,
                text_threshold=0.2, link_threshold=0.15, low_text=0.15, poly=False
            )
            
            print(f"[🔧] 탐지된 박스: {len(boxes)}개")
            
            if len(boxes) == 0:
                print("[❌] 텍스트 영역이 탐지되지 않음")
                return []
            
            # 좌표 조정
            orig_h, orig_w = image_np.shape[:2]
            ratio_w = orig_w / target_width
            ratio_h = orig_h / target_height
            
            boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
            
            # 4. 각 박스에서 텍스트 인식
            text_regions = []
            for i, box in enumerate(boxes):
                if box is None or len(box) != 4:
                    continue
                    
                # 박스 좌표 추출
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                
                # 유효한 크기인지 확인
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    continue
                
                # 이미지 크롭
                cropped = image_np[y1:y2, x1:x2]
                
                # Recognizer로 텍스트 인식
                recognized_text = self._recognize_text_from_crop(cropped)
                
                if recognized_text and recognized_text != "???":
                    detection = [
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        recognized_text,
                        0.8
                    ]
                    text_regions.append(detection)
                    print(f"[✅] Box {i+1}: '{recognized_text}'")
            
            print(f"[✅ Simple Pipeline] {len(text_regions)}개 텍스트 영역 인식됨")
            return text_regions
            
        except Exception as e:
            print(f"[❌ Simple Pipeline] 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _process_with_manual_method(self, image_np):
        """수동 방식 OCR 처리"""
        try:
            import cv2
            
            print("[🚀 ONNX OCR] 이미지 전처리 중...")
            preprocessed = self._preprocess_image_for_onnx(image_np)
            
            print("[🚀 ONNX OCR] 텍스트 탐지 실행 중...")
            # Detector 실행
            detector_outputs = self.easyocr_detector_session.run(None, {"image": preprocessed})
            
            print("[🚀 ONNX OCR] 텍스트 영역 추출 중...")
            # 탐지된 영역 추출
            text_regions = self._postprocess_ocr_result(detector_outputs, image_np.shape)
            
            print(f"[🚀 ONNX OCR] {len(text_regions)}개 영역에서 텍스트 인식 중...")
            # Recognizer로 각 영역의 텍스트 인식
            recognized_results = []
            
            for i, region in enumerate(text_regions):
                bbox, dummy_text, confidence = region
                
                # bbox에서 좌표 추출 (4개 점)
                points = np.array(bbox, dtype=np.int32)
                
                # 패딩을 추가하여 더 큰 영역 추출 (텍스트 주변 여백 포함)
                padding = 10  # 각 방향으로 10픽셀 패딩
                x_min = max(0, np.min(points[:, 0]) - padding)
                y_min = max(0, np.min(points[:, 1]) - padding)
                x_max = min(image_np.shape[1], np.max(points[:, 0]) + padding)
                y_max = min(image_np.shape[0], np.max(points[:, 1]) + padding)
                
                print(f"[🔧 OCR] 영역 {i+1} 크롭: ({x_min},{y_min}) - ({x_max},{y_max}) = {x_max-x_min}x{y_max-y_min}")
                
                # 텍스트 영역 crop
                cropped = image_np[y_min:y_max, x_min:x_max]
                
                if cropped.size == 0:
                    continue
                
                # Recognizer를 위한 전처리
                recognized_text = self._recognize_text_from_crop(cropped)
                
                if recognized_text:
                    # 실제 텍스트로 교체
                    recognized_results.append([bbox, recognized_text, confidence])
                    print(f"[✅ OCR] 영역 {i+1}: {recognized_text}")
            
            return recognized_results
            
        except Exception as e:
            print(f"[❌ ONNX OCR] 처리 실패: {e}")
            return None
    
    def _recognize_text_from_crop(self, cropped_image):
        """실제 EasyOCR 방식으로 cropped 이미지에서 텍스트 인식"""
        try:
            import cv2
            from easyocr.recognition import AlignCollate
            from torch.utils.data import DataLoader, Dataset
            import torch
            import torch.nn.functional as F
            
            # ListDataset을 직접 구현
            class SimpleListDataset(Dataset):
                def __init__(self, img_list):
                    self.img_list = img_list
                
                def __len__(self):
                    return len(self.img_list)
                
                def __getitem__(self, idx):
                    return self.img_list[idx]
            
            # EasyOCR Reader 초기화 (문자셋과 converter 가져오기)
            if not hasattr(self, '_easyocr_reader'):
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                print(f"[🔍 EasyOCR] Reader 초기화 완료 - 문자셋: {len(self._easyocr_reader.character)}개")
            
            # 이미지를 그레이스케일로 변환
            if len(cropped_image.shape) == 3:
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = cropped_image
            
            # PIL Image로 변환 (AlignCollate가 기대하는 형식)
            from PIL import Image
            pil_image = Image.fromarray(gray_image).convert('L')  # 그레이스케일
            
            # ONNX 모델 고정 크기 (1000x64)
            imgH = 64
            imgW = 1000  # ONNX 모델이 기대하는 고정 너비
            h, w = gray_image.shape
            
            print(f"[🔍 EasyOCR] 고정 크기: {imgW}x{imgH} (원본: {w}x{h})")
            
            # AlignCollate로 전처리
            AlignCollate_normal = AlignCollate(
                imgH=imgH, 
                imgW=imgW, 
                keep_ratio_with_pad=True
            )
            
            # PIL Image 리스트로 준비
            img_list = [pil_image]
            test_data = SimpleListDataset(img_list)
            test_loader = DataLoader(
                test_data,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=AlignCollate_normal,
                pin_memory=False,
            )
            
            # 전처리된 이미지 가져오기
            for batch in test_loader:
                # PyTorch Tensor를 NumPy로 변환
                if isinstance(batch, torch.Tensor):
                    normalized = batch.numpy()
                else:
                    # batch가 tuple/list인 경우 첫 번째 요소 사용
                    normalized = batch[0].numpy() if isinstance(batch[0], torch.Tensor) else batch[0]
                
                print(f"[🔍 실제 입력] 형태: {normalized.shape}, 타입: {normalized.dtype}")
                
                # Recognizer 실행
                outputs = self.easyocr_recognizer_session.run(None, {"image": normalized})
                
                # 실제 EasyOCR 방식으로 디코딩
                recognized_text = self._decode_with_easyocr_converter(outputs[0])
                
                return recognized_text
            
            return ""
            
        except Exception as e:
            print(f"[⚠️ Recognizer] EasyOCR 방식 인식 실패: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _decode_with_easyocr_converter(self, output):
        """실제 EasyOCR Converter로 디코딩 (테스트에서 검증된 방식)"""
        try:
            import torch
            import torch.nn.functional as F
            
            # EasyOCR Reader에서 converter 가져오기
            if not hasattr(self, '_easyocr_reader'):
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                print(f"[🔍 EasyOCR] Reader 초기화 완료 - 문자셋: {len(self._easyocr_reader.character)}개")
            
            # 출력을 PyTorch Tensor로 변환
            if isinstance(output, np.ndarray):
                preds = torch.from_numpy(output).float()
            else:
                preds = output
            
            print(f"[🔍 CTC 디코딩] 입력 형태: {preds.shape}")
            
            # Softmax 적용
            preds = F.softmax(preds, dim=2)
            
            # Greedy 디코딩 (테스트에서 검증된 방식)
            _, preds_index = preds.max(2)
            
            print(f"[🔍 CTC 디코딩] 인덱스 형태: {preds_index.shape}")
            
            # 길이 정보 생성
            batch_size = preds.size(0)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            
            # 실제 EasyOCR Converter로 디코딩
            converter = self._easyocr_reader.converter
            preds_str = converter.decode_greedy(
                preds_index.view(-1).data.cpu().detach().numpy(), 
                preds_size.data
            )
            
            decoded_text = preds_str[0] if preds_str else ""
            
            print(f"[✅ CTC 디코딩] 결과: '{decoded_text}'")
            return decoded_text
            
        except Exception as e:
            print(f"[⚠️ EasyOCR Converter] 디코딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return "???"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ultralytics import YOLO
import easyocr
import os
import onnxruntime as ort
import numpy as np

# ONNX 모델 설정
USE_ONNX = True  # True: ONNX 모델 사용, False: Nomic API 사용
ONNX_MODEL_PATH = "C:/Users/csw21/Downloads/nomic_embed_text.onnx/model.onnx/model.onnx"
EASYOCR_ONNX_PATH = "C:/Users/csw21/Downloads/easyocr-easyocrdetector.onnx/model.onnx/model.onnx"
USE_EASYOCR_ONNX = True  # True: EasyOCR ONNX 사용, False: EasyOCR API 사용

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
        self.easyocr_onnx_session = None
        
        # Nomic ONNX 모델 (GPU 우선, CPU 폴백)
        if USE_ONNX and os.path.exists(ONNX_MODEL_PATH):
            self.onnx_session = self._load_onnx_model(ONNX_MODEL_PATH, "Nomic 임베딩")
            if self.onnx_session:
                try:
                    self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                    self._reset_console_color()
                    print("[✅ ONNX] Nomic 모델 로딩 완료!")
                except Exception as e:
                    self._reset_console_color()
                    print(f"[❌ ONNX] Nomic 토크나이저 로딩 실패: {e}")
                    self.onnx_session = None
        
        # EasyOCR ONNX 모델 (GPU 우선, CPU 폴백)
        if USE_EASYOCR_ONNX and os.path.exists(EASYOCR_ONNX_PATH):
            self.easyocr_onnx_session = self._load_onnx_model(EASYOCR_ONNX_PATH, "EasyOCR 탐지")
            self._reset_console_color()  # EasyOCR 로딩 후에도 색상 리셋
        
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
        if self.easyocr_onnx_session:
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
        """ONNX 모델로 이미지에서 텍스트 추출"""
        if not self.easyocr_onnx_session:
            return None
            
        try:
            print("[🚀 ONNX OCR] 이미지 전처리 중...")
            preprocessed = self._preprocess_image_for_onnx(image_np)
            
            print("[🚀 ONNX OCR] 텍스트 탐지 실행 중...")
            # ONNX 모델 실행
            outputs = self.easyocr_onnx_session.run(None, {"image": preprocessed})
            
            print("[🚀 ONNX OCR] 결과 후처리 중...")
            # 결과 후처리
            detections = self._postprocess_ocr_result(outputs, image_np.shape)
            
            return detections
            
        except Exception as e:
            print(f"[❌ ONNX OCR] 처리 실패: {e}")
            return None
    
    def load_summarizer(self):
        """요약 모델 로딩 (비활성화 - Qwen 사용)"""
        print("[ℹ️ BART 요약 모델 비활성화됨 - Qwen 사용]")
        return False  # 항상 False 반환하여 Qwen 사용 강제
    
    
    def _get_embeddings(self, texts):
        """텍스트 임베딩 생성 (ONNX 우선, API 폴백)"""
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
        
        # Nomic API 사용
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
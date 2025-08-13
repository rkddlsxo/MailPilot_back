import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ultralytics import YOLO
import easyocr
from nomic import embed, login
import os

class AIModels:
    def __init__(self, config):
        self.config = config
        
        # 모델 인스턴스들
        self.yolo_model = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.ocr_reader = None
        self.summarizer = None
        
        # Nomic 로그인
        login(token=config.NOMIC_TOKEN)
        print("[✅ Nomic 로그인 완료]")
    
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
        """Qwen 모델 로딩"""
        if self.qwen_model is None:
            print("[🤖 Qwen 모델 로딩 시작]")
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.QWEN_MODEL, 
                    trust_remote_code=True
                )
                # Meta Device 오류 방지를 위한 안전한 로딩
                if torch.cuda.is_available():
                    # GPU 사용 가능한 경우
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(
                        self.config.QWEN_MODEL,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        device_map=None  # auto 대신 None 사용
                    )
                    self.qwen_model.to("cuda")
                else:
                    # CPU만 사용하는 경우 (Meta Device 문제 방지)
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(
                        self.config.QWEN_MODEL,
                        torch_dtype=torch.float32,  # CPU에서는 float32 필수
                        trust_remote_code=True,
                        low_cpu_mem_usage=True  # 메모리 절약
                    )
                    self.qwen_model.to("cpu")
                self.qwen_model.eval()
                print("[✅ Qwen 모델 로딩 완료]")
                return True
            except Exception as e:
                print(f"[❗Qwen 모델 로딩 실패] {str(e)}")
                return False
        return True
    
    def load_ocr_model(self):
        """OCR 모델 로딩"""
        if self.ocr_reader is None:
            try:
                print("[📖 EasyOCR 모델 로딩 시작]")
                self.ocr_reader = easyocr.Reader(['ko', 'en'])
                print("[✅ EasyOCR 모델 로딩 완료]")
                return True
            except Exception as e:
                print(f"[❗EasyOCR 모델 로딩 실패] {str(e)}")
                return False
        return True
    
    def load_summarizer(self):
        """요약 모델 로딩 (비활성화 - Qwen 사용)"""
        print("[ℹ️ BART 요약 모델 비활성화됨 - Qwen 사용]")
        return False  # 항상 False 반환하여 Qwen 사용 강제
    
    
    def classify_email(self, text):
        """이메일 분류"""
        try:
            text_inputs = [text] + self.config.CANDIDATE_LABELS
            result = embed.text(text_inputs, model='nomic-embed-text-v1', task_type='classification')
            
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ultralytics import YOLO
import easyocr
from huggingface_hub import InferenceClient
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
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    self.config.QWEN_MODEL,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
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
        """요약 모델 로딩"""
        if self.summarizer is None:
            try:
                self.summarizer = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn", 
                    tokenizer="facebook/bart-large-cnn"
                )
                print("[✅ 요약 모델 로딩 완료]")
                return True
            except Exception as e:
                print(f"[❗요약 모델 로딩 실패] {str(e)}")
                return False
        return True
    
    def get_inference_client(self):
        """HuggingFace Inference Client 생성"""
        return InferenceClient(
            model=self.config.HUGGINGFACE_MODEL,
            token=self.config.HF_TOKEN
        )
    
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
#!/usr/bin/env python3
# Test ONNX integration

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ai_models import AIModels
from config.config import Config

def test_onnx_models():
    print("[🧪 ONNX 통합 테스트 시작]")
    
    # Config 로드
    config = Config()
    
    # AI Models 초기화
    ai_models = AIModels(config)
    
    # Nomic ONNX 모델 테스트
    if ai_models.onnx_session and ai_models.bert_tokenizer:
        print("[LOG]")
        
        try:
            # 샘플 텍스트로 임베딩 생성 테스트
            test_texts = [LOG]
            result = ai_models._get_embeddings(test_texts)
            
            if result and 'embeddings' in result:
                print(f"[LOG] {len(result[LOG])}개 임베딩")
                print(f"[📊 임베딩 차원] {len(result[LOG][LOG])}")
            else:
                print("[LOG]")
                
        except Exception as e:
            print(f"[LOG] {e}")
    else:
        print("[LOG]")
    
    # EasyOCR ONNX 모델 테스트
    if ai_models.easyocr_onnx_session:
        print("[LOG]")
    else:
        print("[LOG]")
    
    print("[🧪 ONNX 통합 테스트 완료]")

if __name__ == "__main__":
    test_onnx_models()
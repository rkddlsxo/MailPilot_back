# services/attachment_service.py - 첨부파일 처리 서비스

import os
import io
import tempfile
import hashlib
from pathlib import Path
import numpy as np

# 선택적 임포트 - 없는 라이브러리는 비활성화
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[⚠️ PIL/Pillow 없음 - 이미지 처리 비활성화]")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("[⚠️ pdfplumber 없음 - PDF 처리 비활성화]")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("[⚠️ PyPDF2 없음 - PDF 백업 처리 비활성화]")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("[⚠️ python-docx 없음 - Word 문서 처리 비활성화]")

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    print("[⚠️ python-pptx 없음 - PowerPoint 처리 비활성화]")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[⚠️ pandas 없음 - Excel 처리 비활성화]")

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("[⚠️ pdf2image 없음 - PDF OCR 처리 비활성화]")

class AttachmentService:
    def __init__(self, config, ai_models):
        self.config = config
        self.ai_models = ai_models
        self.attachment_cache = {}
        
        # 사용 가능한 기능 체크
        self.features = {
            'image_processing': PIL_AVAILABLE,
            'pdf_processing': PDFPLUMBER_AVAILABLE or PYPDF2_AVAILABLE,
            'docx_processing': DOCX_AVAILABLE,
            'pptx_processing': PPTX_AVAILABLE,
            'xlsx_processing': PANDAS_AVAILABLE,
            'pdf_ocr': PDF2IMAGE_AVAILABLE,
            'yolo': hasattr(ai_models, 'yolo_available') and ai_models.yolo_available,
            'ocr': hasattr(ai_models, 'ocr_available') and ai_models.ocr_available
        }
        
        print(f"[📎 첨부파일 서비스 초기화] 사용 가능한 기능: {sum(self.features.values())}/{len(self.features)}")
    
    def process_email_attachments(self, email_message, email_subject, email_id):
        """이메일에서 첨부파일을 추출하고 처리 (캐싱 포함)"""
        cache_key = f"email_{email_id}"
        
        # 캐시 확인
        if cache_key in self.attachment_cache:
            print(f"[📎 캐시 사용] {email_subject[:30]}...")
            return self.attachment_cache[cache_key]
        
        attachments = []
        print(f"[📎 새로운 첨부파일 처리] {email_subject[:30]}...")
        
        try:
            for part in email_message.walk():
                if part.get_content_disposition() == 'attachment':
                    attachment_info = self._process_single_attachment(part, email_subject)
                    if attachment_info:
                        attachments.append(attachment_info)
        except Exception as e:
            print(f"[❗첨부파일 워킹 오류] {str(e)}")
        
        # 캐시 저장
        self.attachment_cache[cache_key] = attachments
        self._manage_cache_size()
        
        print(f"[✅ 첨부파일 처리 완료] {len(attachments)}개 처리됨")
        return attachments
    
    def _process_single_attachment(self, part, email_subject):
        """개별 첨부파일 처리"""
        try:
            filename = self._decode_filename(part.get_filename())
            if not filename:
                return None
            
            attachment_data = part.get_payload(decode=True)
            if not attachment_data:
                return None
            
            file_ext = Path(filename).suffix.lower()
            mime_type = part.get_content_type()
            
            attachment_info = {
                'filename': filename,
                'size': len(attachment_data),
                'mime_type': mime_type,
                'extension': file_ext
            }
            
            # 파일 타입별 처리
            if file_ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}:
                attachment_info.update(self._process_image(attachment_data, filename))
            elif file_ext == '.pdf' or 'pdf' in mime_type:
                attachment_info.update(self._process_pdf(attachment_data, filename))
            elif file_ext == '.docx' or 'wordprocessingml' in mime_type:
                attachment_info.update(self._process_docx(attachment_data, filename))
            elif file_ext == '.pptx' or 'presentationml' in mime_type:
                attachment_info.update(self._process_pptx(attachment_data, filename))
            elif file_ext in ['.xlsx', '.xls'] or 'spreadsheetml' in mime_type:
                attachment_info.update(self._process_xlsx(attachment_data, filename))
            else:
                attachment_info.update({'type': 'other', 'processing_method': 'metadata_only'})
            
            return attachment_info
            
        except Exception as e:
            print(f"[❗첨부파일 처리 오류] {filename if 'filename' in locals() else 'Unknown'}: {str(e)}")
            return None
    
    def _decode_filename(self, filename):
        """파일명 디코딩"""
        if not filename:
            return None
        
        try:
            from email.header import decode_header
            decoded_parts = decode_header(filename)
            if decoded_parts and decoded_parts[0]:
                decoded_filename = decoded_parts[0]
                if isinstance(decoded_filename[0], bytes):
                    return decoded_filename[0].decode(decoded_filename[1] or 'utf-8')
                else:
                    return decoded_filename[0]
        except:
            pass
        
        return filename
    
    def _process_image(self, attachment_data, filename):
        """이미지 처리 (YOLO + OCR)"""
        try:
            if not PIL_AVAILABLE:
                return {'type': 'image', 'error': 'PIL not available', 'processing_method': 'disabled'}
            
            # YOLO 객체 인식
            yolo_detections = []
            if self.features['yolo'] and self.ai_models.load_yolo_model():
                yolo_detections = self._yolo_detect_objects(attachment_data)
            
            # OCR 텍스트 추출
            ocr_result = {'text': '', 'success': False}
            if self.features['ocr'] and self.ai_models.load_ocr_model():
                ocr_result = self._extract_text_with_ocr(attachment_data, filename)
            
            result = {
                'type': 'image',
                'yolo_detections': yolo_detections,
                'detected_objects': [det['class'] for det in yolo_detections],
                'object_count': len(yolo_detections),
                'extracted_text': ocr_result.get('text', ''),
                'ocr_success': ocr_result.get('success', False),
                'processing_method': f"YOLO({len(yolo_detections)}) + OCR({ocr_result.get('success', False)})"
            }
            
            # 텍스트 요약 생성
            if ocr_result.get('success') and ocr_result.get('text'):
                result['text_summary'] = self._summarize_document(
                    ocr_result['text'], filename, 'image_with_text'
                )
            
            return result
            
        except Exception as e:
            print(f"[❗이미지 처리 오류] {str(e)}")
            return {'type': 'image', 'error': str(e), 'processing_method': 'failed'}
    
    def _yolo_detect_objects(self, image_data):
        """YOLO 객체 인식"""
        try:
            if not PIL_AVAILABLE:
                return []
            
            # 이미지 로드 및 전처리
            image = Image.open(io.BytesIO(image_data))
            
            # RGBA → RGB 변환
            if image.mode in ['RGBA', 'LA']:
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image)
            
            # YOLO 추론
            results = self.ai_models.yolo_model(image_np, conf=0.2)
            
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    class_name = self.ai_models.yolo_model.names[cls]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'class_id': cls
                    })
            
            return detections
            
        except Exception as e:
            print(f"[❗YOLO 처리 오류] {str(e)}")
            return []
    
    def _extract_text_with_ocr(self, attachment_data, filename):
        """OCR 텍스트 추출"""
        try:
            if not PIL_AVAILABLE:
                return {'text': '', 'success': False, 'error': 'PIL not available'}
            
            image = Image.open(io.BytesIO(attachment_data))
            
            # 이미지 전처리
            if image.mode in ['RGBA', 'LA']:
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_np = np.array(image)
            
            # OCR 수행
            result = self.ai_models.ocr_reader.readtext(image_np, paragraph=True)
            
            text = ""
            for detection in result:
                if len(detection) >= 3:
                    text_content = detection[1]
                    confidence = detection[2]
                    if confidence > 0.5:
                        text += text_content + " "
            
            return {
                'text': text.strip(),
                'success': bool(text.strip()),
                'method': 'ocr'
            }
            
        except Exception as e:
            print(f"[❗OCR 오류] {str(e)}")
            return {'text': '', 'success': False, 'error': str(e)}
    
    def _process_pdf(self, attachment_data, filename):
        """PDF 처리"""
        if not PDFPLUMBER_AVAILABLE and not PYPDF2_AVAILABLE:
            return {'type': 'document_pdf', 'error': 'PDF libraries not available', 'extraction_success': False}
        
        try:
            # pdfplumber로 텍스트 추출 시도
            if PDFPLUMBER_AVAILABLE:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(attachment_data)
                    temp_file_path = temp_file.name
                
                try:
                    with pdfplumber.open(temp_file_path) as pdf:
                        text = ""
                        for page_num, page in enumerate(pdf.pages):
                            page_text = page.extract_text()
                            if page_text:
                                text += f"\n=== 페이지 {page_num + 1} ===\n{page_text}\n"
                    
                    if text.strip():
                        result = {
                            'type': 'document_pdf',
                            'extracted_text': text.strip(),
                            'extraction_success': True,
                            'extraction_method': 'pdfplumber',
                            'pages': len(pdf.pages)
                        }
                        
                        # 문서 요약 생성
                        result['document_summary'] = self._summarize_document(
                            text, filename, 'PDF 보고서'
                        )
                        
                        return result
                        
                except Exception as e:
                    print(f"[⚠️ pdfplumber 실패] {str(e)}")
                finally:
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
            
            # PyPDF2로 재시도
            if PYPDF2_AVAILABLE:
                return self._process_pdf_fallback(attachment_data, filename)
            
            return {'type': 'document_pdf', 'extraction_success': False, 'error': 'No PDF library available'}
            
        except Exception as e:
            print(f"[❗PDF 처리 오류] {str(e)}")
            return {'type': 'document_pdf', 'error': str(e), 'extraction_success': False}
    
    def _process_pdf_fallback(self, attachment_data, filename):
        """PDF 대체 처리 (PyPDF2)"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(attachment_data)
                temp_file_path = temp_file.name
            
            with open(temp_file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n=== 페이지 {page_num + 1} ===\n{page_text}\n"
            
            if text.strip():
                result = {
                    'type': 'document_pdf',
                    'extracted_text': text.strip(),
                    'extraction_success': True,
                    'extraction_method': 'pypdf2',
                    'pages': len(pdf_reader.pages)
                }
                
                result['document_summary'] = self._summarize_document(
                    text, filename, 'PDF 보고서'
                )
                
                return result
            
            return {'type': 'document_pdf', 'extraction_success': False}
            
        except Exception as e:
            return {'type': 'document_pdf', 'error': str(e), 'extraction_success': False}
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _process_docx(self, attachment_data, filename):
        """Word 문서 처리"""
        if not DOCX_AVAILABLE:
            return {'type': 'document_word', 'error': 'python-docx not available', 'extraction_success': False}
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                temp_file.write(attachment_data)
                temp_file_path = temp_file.name
            
            doc = Document(temp_file_path)
            
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            # 표 내용도 추출
            for table in doc.tables:
                text += "\n=== 표 데이터 ===\n"
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text += " | ".join(row_text) + "\n"
            
            if text.strip():
                result = {
                    'type': 'document_word',
                    'extracted_text': text.strip(),
                    'extraction_success': True,
                    'paragraphs': len(doc.paragraphs),
                    'tables': len(doc.tables)
                }
                
                result['document_summary'] = self._summarize_document(
                    text, filename, 'Word 문서'
                )
                
                return result
            
            return {'type': 'document_word', 'extraction_success': False}
            
        except Exception as e:
            return {'type': 'document_word', 'error': str(e), 'extraction_success': False}
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _process_pptx(self, attachment_data, filename):
        """PowerPoint 처리"""
        if not PPTX_AVAILABLE:
            return {'type': 'document_presentation', 'error': 'python-pptx not available', 'extraction_success': False}
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as temp_file:
                temp_file.write(attachment_data)
                temp_file_path = temp_file.name
            
            prs = Presentation(temp_file_path)
            
            text = ""
            for slide_num, slide in enumerate(prs.slides):
                text += f"\n=== 슬라이드 {slide_num + 1} ===\n"
                
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text += shape.text + "\n"
                    
                    if hasattr(shape, 'has_table') and shape.has_table:
                        text += "\n--- 표 ---\n"
                        table = shape.table
                        for row in table.rows:
                            row_text = []
                            for cell in row.cells:
                                if cell.text.strip():
                                    row_text.append(cell.text.strip())
                            if row_text:
                                text += " | ".join(row_text) + "\n"
            
            if text.strip():
                result = {
                    'type': 'document_presentation',
                    'extracted_text': text.strip(),
                    'extraction_success': True,
                    'slides': len(prs.slides)
                }
                
                result['document_summary'] = self._summarize_document(
                    text, filename, 'PowerPoint 프레젠테이션'
                )
                
                return result
            
            return {'type': 'document_presentation', 'extraction_success': False}
            
        except Exception as e:
            return {'type': 'document_presentation', 'error': str(e), 'extraction_success': False}
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _process_xlsx(self, attachment_data, filename):
        """Excel 처리"""
        if not PANDAS_AVAILABLE:
            return {'type': 'document_spreadsheet', 'error': 'pandas not available', 'extraction_success': False}
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
                temp_file.write(attachment_data)
                temp_file_path = temp_file.name
            
            xl_file = pd.ExcelFile(temp_file_path)
            
            text = ""
            total_rows = 0
            
            for sheet_name in xl_file.sheet_names:
                df = pd.read_excel(temp_file_path, sheet_name=sheet_name)
                
                if not df.empty:
                    text += f"\n=== 시트: {sheet_name} ===\n"
                    text += "컬럼: " + " | ".join(str(col) for col in df.columns) + "\n\n"
                    
                    for idx, row in df.head(20).iterrows():
                        row_text = []
                        for value in row:
                            if pd.notna(value):
                                row_text.append(str(value))
                            else:
                                row_text.append("")
                        text += " | ".join(row_text) + "\n"
                    
                    total_rows += len(df)
                    
                    if len(df) > 20:
                        text += f"... (총 {len(df)}행 중 처음 20행만 표시)\n"
            
            if text.strip():
                result = {
                    'type': 'document_spreadsheet',
                    'extracted_text': text.strip(),
                    'extraction_success': True,
                    'sheets': len(xl_file.sheet_names),
                    'total_rows': total_rows
                }
                
                result['document_summary'] = self._summarize_document(
                    text, filename, 'Excel 스프레드시트'
                )
                
                return result
            
            return {'type': 'document_spreadsheet', 'extraction_success': False}
            
        except Exception as e:
            return {'type': 'document_spreadsheet', 'error': str(e), 'extraction_success': False}
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _summarize_document(self, text, filename, file_type):
        """문서 요약 생성"""
        try:
            if len(text) > 4000:
                text = text[:4000] + "..."
            
            if not self.config.HF_TOKEN:
                return text[:300] + "..." if len(text) > 300 else text
            
            try:
                client = self.ai_models.get_inference_client()
                
                prompt = f"""다음은 '{filename}' 파일의 내용입니다. 이 문서를 요약해주세요.

파일 형식: {file_type}
내용:
{text}

요약 지침:
1. 주요 내용을 3-5개 포인트로 요약
2. 핵심 키워드와 수치 포함
3. 150자 이내로 간결하게
4. 한국어로 응답

요약:"""
                
                messages = [
                    {"role": "system", "content": "당신은 문서 요약 전문가입니다."},
                    {"role": "user", "content": prompt}
                ]
                
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=200,
                    temperature=0.3
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception:
                # 간단한 요약으로 fallback
                sentences = text.split('.')
                important_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
                return '. '.join(important_sentences) + '.' if important_sentences else text[:200] + "..."
                
        except Exception as e:
            return text[:200] + "..." if len(text) > 200 else text
    
    def _manage_cache_size(self):
        """캐시 크기 관리"""
        if len(self.attachment_cache) > self.config.MAX_CACHE_SIZE:
            oldest_key = next(iter(self.attachment_cache))
            del self.attachment_cache[oldest_key]
            print(f"[🗑️ 캐시 정리] 오래된 항목 삭제: {oldest_key}")
    
    def generate_attachment_summary(self, attachments):
        """첨부파일 요약 생성"""
        if not attachments:
            return ""
        
        total_files = len(attachments)
        
        # 파일 타입별 분류
        images = [att for att in attachments if att.get('type') == 'image']
        documents = [att for att in attachments if att.get('type', '').startswith('document_')]
        others = [att for att in attachments if att.get('type') not in ['image'] and not att.get('type', '').startswith('document_')]
        
        summary_parts = []
        
        if images:
            total_objects = sum(att.get('object_count', 0) for att in images)
            ocr_texts = [att for att in images if att.get('ocr_success')]
            
            if total_objects > 0:
                summary_parts.append(f"이미지 {len(images)}개({total_objects}개 객체)")
            else:
                summary_parts.append(f"이미지 {len(images)}개")
                
            if ocr_texts:
                summary_parts.append(f"텍스트 추출 {len(ocr_texts)}개")
        
        if documents:
            doc_types = {}
            successful_extractions = 0
            
            for doc in documents:
                doc_type = doc.get('type', '').replace('document_', '')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                if doc.get('extraction_success'):
                    successful_extractions += 1
            
            for doc_type, count in doc_types.items():
                type_names = {
                    'pdf': 'PDF', 
                    'word': 'Word', 
                    'presentation': 'PPT', 
                    'spreadsheet': 'Excel'
                }
                type_name = type_names.get(doc_type, doc_type.upper())
                summary_parts.append(f"{type_name} {count}개")
            
            if successful_extractions > 0:
                summary_parts.append(f"요약 가능 {successful_extractions}개")
        
        if others:
            summary_parts.append(f"기타 {len(others)}개")
        
        if summary_parts:
            return f"📎 {total_files}개 파일: " + ", ".join(summary_parts)
        else:
            return f"📎 {total_files}개 파일"
    
    def clear_cache(self):
        """캐시 초기화"""
        cache_count = len(self.attachment_cache)
        self.attachment_cache.clear()
        return cache_count
    
    def get_available_features(self):
        """사용 가능한 기능 목록 반환"""
        return self.features
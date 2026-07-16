import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import re
import json
from datetime import datetime
import os
import logging
from pathlib import Path
import sys

try:
    from pypdf import PdfWriter
except ImportError:
    PdfWriter = None

try:
    from docx2pdf import convert as docx_convert
except ImportError:
    docx_convert = None

# Configure Tesseract path (Windows fallback)
if os.name == 'nt':
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrademarkOppositionOCR:
    """Extract structured data from TM-5 Notice of Opposition forms"""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.raw_text = ""
        self.extracted_data = {}
        
    def preprocess_image(self):
        """Enhance image for better OCR results"""
        logging.info(f"Preprocessing image: {self.image_path}")
        img = cv2.imread(str(self.image_path))
        if img is None:
            raise ValueError(f"Could not read image: {self.image_path}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(thresh, h=30)
        
        coords = np.column_stack(np.where(denoised > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            if abs(angle) > 0.5:
                (h, w) = denoised.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                denoised = cv2.warpAffine(denoised, M, (w, h), 
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
        
        pil_img = Image.fromarray(denoised)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2.0)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        
        return pil_img
    
    def clean_ocr_text(self, text):
        """Clean extracted text"""
        text = re.sub(r'\n{3,}', '\n\n', text)
        replacements = {'0': 'O', '1': 'I', '5': 'S'}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def extract_text(self):
        """Extract raw text using Tesseract OCR"""
        processed_img = self.preprocess_image()
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./:-_@#()"\' ,'
        
        self.raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
        self.raw_text = self.clean_ocr_text(self.raw_text)
        return self.raw_text
    
    def parse_opponent_info(self):
        text = self.raw_text
        data = {
            'opponent': {
                'name': self._extract_between(text, 'Name & address of Opponent', 'Trademark Application'),
                'address': self._extract_between(text, 'Name & address of Opponent', 'Trademark Application'),
                'trademark_application_no': self._extract_field(text, r'Application No\.?:\s*(\d+)'),
                'filing_date': self._extract_field(text, r'Date of filing:\s*(\d+/\d+/\d+)'),
                'mark': self._extract_field(text, r'Trademark applied for:\s*([^\n]+(?:\n[^\n]+)*?)(?=Classes|Class)'),
                'class': self._extract_field(text, r'Classes?:\s*(\d+)'),
                'goods': self._extract_field(text, r'Goods & Services\s*([^\n]+(?:\n[^\n]+)*?)(?=Basis of Opposition|Name & address)'),
                'basis_of_opposition': self._extract_field(text, r'Basis of Opposition:\s*([^\n]+)'),
                'agent': self._extract_field(text, r'Name & address of Opponent\'?s agent:\s*([^\n]+(?:\n[^\n]+)*?)(?=APPLICANT|$)')
            }
        }
        
        if data['opponent']['name']:
            name_addr = data['opponent']['name'].strip().split('\n')
            if len(name_addr) >= 2:
                data['opponent']['name'] = name_addr[0].strip()
                data['opponent']['address'] = '\n'.join(name_addr[1:]).strip()
        
        return data
    
    def parse_applicant_info(self):
        text = self.raw_text
        data = {
            'applicant': {
                'name': self._extract_between(text, 'Name & address of Applicant:', 'Application No'),
                'address': self._extract_between(text, 'Name & address of Applicant:', 'Application No'),
                'application_no': self._extract_field(text, r'Application No\.?:\s*(\d+)'),
                'filing_date': self._extract_field(text, r'Date of Filing:\s*(\d+/\d+/\d+)'),
                'mark': self._extract_field(text, r'Trademark applied for:\s*([^\n]+(?:\n[^\n]+)*?)(?=Class|\d)'),
                'class': self._extract_field(text, r'Class\s*(\d+)'),
                'goods': self._extract_field(text, r'Goods & Services\s*([^\n]+(?:\n[^\n]+)*?)(?=We,|$)')
            }
        }
        
        if data['applicant']['name']:
            name_addr = data['applicant']['name'].strip().split('\n')
            if len(name_addr) >= 2:
                data['applicant']['name'] = name_addr[0].strip()
                data['applicant']['address'] = '\n'.join(name_addr[1:]).strip()
        
        return data
    
    def parse_grounds(self):
        text = self.raw_text
        sections = re.findall(r'S[. ]*(\d+)[)]?', text)
        grounds = {
            'sections': list(set(sections)),
            'similarity_claims': bool(re.search(r'similar|identical|replica|imitat|counterfeit', text, re.IGNORECASE)),
            'dilution_claims': bool(re.search(r'dilution|well-known', text, re.IGNORECASE)),
            'bad_faith_claims': bool(re.search(r'mala fide|dishonest|malicious', text, re.IGNORECASE)),
            'confusion_claims': bool(re.search(r'confusion|deceive|mislead', text, re.IGNORECASE)),
            'prior_use_claims': bool(re.search(r'prior user|original|adopter|earlier', text, re.IGNORECASE))
        }
        return grounds
    
    def parse_prayer(self):
        text = self.raw_text
        prayer_text = self._extract_between(text, 'PRAYER', 'Our Address for service')
        if not prayer_text:
            prayer_text = self._extract_between(text, 'PRAYER', 'Dated')
        
        requests = []
        if prayer_text:
            requests = re.findall(r'[a-z]\.?\s*([^\n]+)', prayer_text)
            if not requests:
                requests = [line.strip() for line in prayer_text.split('\n') if line.strip()]
        
        return {'prayer': prayer_text, 'requests': requests}
    
    def parse_opposition_number(self):
        text = self.raw_text
        return self._extract_field(text, r'Opposition #?\s*(\d+)/2025')
    
    def _extract_field(self, text, pattern):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_between(self, text, start, end):
        try:
            pattern = f'{start}(.*?){end}'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        except:
            pass
        return None
    
    def extract_all(self):
        logging.info("Extracting text from image...")
        self.extract_text()
        
        logging.info("Parsing document structure...")
        
        data = {
            'metadata': {
                'extraction_date': datetime.now().isoformat(),
                'source_file': os.path.basename(self.image_path),
                'ocr_engine': 'Tesseract'
            },
            'opposition_number': self.parse_opposition_number(),
            'opponent': self.parse_opponent_info()['opponent'],
            'applicant': self.parse_applicant_info()['applicant'],
            'grounds': self.parse_grounds(),
            'prayer': self.parse_prayer(),
            'raw_text': self.raw_text
        }
        
        self.extracted_data = data
        return data
    
    def validate_extraction(self):
        """Validate if critical fields were extracted"""
        critical_fields = [
            'opponent.name',
            'opponent.trademark_application_no',
            'applicant.name',
            'applicant.application_no',
            'grounds.sections'
        ]
        missing = []
        for field in critical_fields:
            parts = field.split('.')
            value = self.extracted_data
            try:
                for part in parts:
                    value = value.get(part, {})
                if not value:
                    missing.append(field)
            except:
                missing.append(field)
        return len(missing) == 0, missing

    def save_to_json(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.extracted_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Data saved to {output_path}")
    
    def save_to_markdown(self, output_path):
        data = self.extracted_data
        
        requests_str = ''.join(['- ' + req + '\n' for req in data['prayer'].get('requests', [])])
        
        md = f"""# TRADEMARK OPPOSITION EXTRACTION REPORT

**Extraction Date:** {data['metadata']['extraction_date']}  
**Source File:** {data['metadata']['source_file']}  
**OCR Engine:** {data['metadata']['ocr_engine']}

---

## OPPOSITION DETAILS
- **Opposition Number:** {data.get('opposition_number', 'Not found')}

---

## OPPONENT/OBJECTOR

| Field | Value |
|-------|-------|
| **Name** | {data['opponent'].get('name', 'Not found')} |
| **Address** | {data['opponent'].get('address', 'Not found')} |
| **TM Application No.** | {data['opponent'].get('trademark_application_no', 'Not found')} |
| **Filing Date** | {data['opponent'].get('filing_date', 'Not found')} |
| **Trademark** | {data['opponent'].get('mark', 'Not found')} |
| **Class** | {data['opponent'].get('class', 'Not found')} |
| **Goods/Services** | {str(data['opponent'].get('goods', 'Not found'))[:200]}... |
| **Basis of Opposition** | {data['opponent'].get('basis_of_opposition', 'Not found')} |
| **Agent** | {data['opponent'].get('agent', 'Not found')} |

---

## APPLICANT (RESPONDENT)

| Field | Value |
|-------|-------|
| **Name** | {data['applicant'].get('name', 'Not found')} |
| **Address** | {data['applicant'].get('address', 'Not found')} |
| **Application No.** | {data['applicant'].get('application_no', 'Not found')} |
| **Filing Date** | {data['applicant'].get('filing_date', 'Not found')} |
| **Trademark** | {data['applicant'].get('mark', 'Not found')} |
| **Class** | {data['applicant'].get('class', 'Not found')} |
| **Goods/Services** | {str(data['applicant'].get('goods', 'Not found'))[:200]}... |

---

## GROUNDS OF OPPOSITION

| Ground | Present |
|--------|---------|
| **Similarity/Replica Claims** | {data['grounds'].get('similarity_claims', False)} |
| **Dilution Claims** | {data['grounds'].get('dilution_claims', False)} |
| **Bad Faith Claims** | {data['grounds'].get('bad_faith_claims', False)} |
| **Confusion/Deception Claims** | {data['grounds'].get('confusion_claims', False)} |
| **Prior Use Claims** | {data['grounds'].get('prior_use_claims', False)} |

**Legal Sections Cited:** {', '.join(data['grounds'].get('sections', []))}

---

## PRAYER/RELIEF REQUESTED

{data['prayer'].get('prayer', 'Not found')}

### Specific Requests:
{requests_str}
---

## RAW TEXT (First 500 chars)
{str(data.get('raw_text', ''))[:500]}...

---

*Generated automatically by Trademark Opposition OCR System*
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        logging.info(f"Markdown report saved to {output_path}")

def batch_process_opposition_documents(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    extensions = {'.jpeg', '.jpg', '.png', '.tiff', '.bmp'}
    
    for file in input_path.glob('*'):
        if file.suffix.lower() in extensions:
            logging.info(f"Processing: {file.name}")
            
            try:
                ocr = TrademarkOppositionOCR(str(file))
                data = ocr.extract_all()
                is_valid, missing = ocr.validate_extraction()
                
                if not is_valid:
                    logging.warning(f"Validation failed for {file.name}. Missing fields: {missing}")
                
                base_name = file.stem
                ocr.save_to_json(output_path / f'{base_name}_data.json')
                ocr.save_to_markdown(output_path / f'{base_name}_report.md')
                
                results.append(data)
            except Exception as e:
                logging.error(f"Error processing {file.name}: {e}")
    
    master_data = {
        'total_documents': len(results),
        'processed_files': [r['metadata']['source_file'] for r in results],
        'extraction_date': datetime.now().isoformat()
    }
    
    master_summary_path = output_path / 'master_summary.json'
    with open(master_summary_path, 'w', encoding='utf-8') as f:
        json.dump(master_data, f, indent=2)
    
    logging.info(f"Processed {len(results)} documents successfully. Master summary saved to {master_summary_path}")
    return results

def combine_images_to_pdf(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    extensions = {'.jpeg', '.jpg', '.png', '.tiff', '.bmp'}
    
    images = []
    for file in input_path.glob('*'):
        if file.suffix.lower() in extensions:
            img = Image.open(file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            
    if not images:
        print("No images found in pdf_input folder.")
        return
        
    output_file = output_path / f"combined_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    images[0].save(output_file, save_all=True, append_images=images[1:])
    print(f"✅ Successfully combined {len(images)} images into {output_file}")

def combine_pdfs(input_folder, output_folder):
    if PdfWriter is None:
        print("Error: 'pypdf' library is not installed. Please run: pip install pypdf")
        return
        
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    merger = PdfWriter()
    pdf_files = list(input_path.glob('*.pdf'))
    
    if not pdf_files:
        print("No PDF files found in pdf_input folder.")
        return
        
    for pdf in pdf_files:
        merger.append(str(pdf))
        
    output_file = output_path / f"combined_pdfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    merger.write(str(output_file))
    merger.close()
    print(f"✅ Successfully combined {len(pdf_files)} PDFs into {output_file}")

def combine_docs_to_pdf(input_folder, output_folder):
    if docx_convert is None or PdfWriter is None:
        print("Error: 'docx2pdf' and/or 'pypdf' libraries are missing. Please run: pip install docx2pdf pypdf")
        return
        
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    doc_files = list(input_path.glob('*.doc')) + list(input_path.glob('*.docx'))
    
    if not doc_files:
        print("No Word documents found in pdf_input folder.")
        return
        
    temp_dir = output_path / "temp_pdf_conversion"
    temp_dir.mkdir(exist_ok=True)
    
    merger = PdfWriter()
    
    try:
        for doc in doc_files:
            pdf_path = temp_dir / f"{doc.stem}.pdf"
            print(f"Converting {doc.name} to PDF...")
            
            if doc.suffix.lower() == '.doc':
                try:
                    import win32com.client
                    word = win32com.client.Dispatch('Word.Application')
                    # FileFormat 17 is wdFormatPDF
                    doc_obj = word.Documents.Open(str(doc.absolute()))
                    doc_obj.SaveAs(str(pdf_path.absolute()), FileFormat=17)
                    doc_obj.Close()
                    word.Quit()
                except Exception as e:
                    print(f"Failed to convert {doc.name}: {e}")
                    continue
            else:
                docx_convert(str(doc), str(pdf_path))
                
            if pdf_path.exists():
                merger.append(str(pdf_path))
            
        output_file = output_path / f"combined_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        merger.write(str(output_file))
        merger.close()
        print(f"✅ Successfully combined {len(doc_files)} documents into {output_file}")
    finally:
        for temp_file in temp_dir.glob('*.pdf'):
            try:
                temp_file.unlink()
            except Exception:
                pass
        try:
            temp_dir.rmdir()
        except Exception:
            pass

def batch_pdf_menu():
    print("\n--- Batch PDF Combiner ---")
    print("Select input file type to combine:")
    print("1. Images (.jpg, .png, etc.)")
    print("2. PDF (.pdf)")
    print("3. Word Documents (.doc, .docx)")
    
    choice = input("Select an option (1/2/3): ").strip()
    
    input_dir = 'pdf_input'
    output_dir = 'pdf_output'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if choice == '1':
        combine_images_to_pdf(input_dir, output_dir)
    elif choice == '2':
        combine_pdfs(input_dir, output_dir)
    elif choice == '3':
        combine_docs_to_pdf(input_dir, output_dir)
    else:
        print("Invalid choice. Returning to main menu.")

if __name__ == '__main__':
    while True:
        print("\n=== Brandex-OCR Utility ===")
        print("1. Extract OCR from Images")
        print("2. Create Combined Batch PDF")
        print("3. Exit")
        choice = input("Select an option (1/2/3): ").strip()
        
        if choice == '1':
            input_dir = 'input'
            output_dir = 'output'
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            logging.info("Starting batch processing...")
            batch_process_opposition_documents(input_dir, output_dir)
        elif choice == '2':
            batch_pdf_menu()
        elif choice == '3':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice, please select 1, 2, or 3.")

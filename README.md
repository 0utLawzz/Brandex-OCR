# Brandex-OCR

Brandex-OCR is a Python-based utility to extract structured data from TM-5 Notice of Opposition forms utilizing Tesseract OCR.

## Prerequisites

1. Install Tesseract OCR on your system.
2. Install Python requirements:
   ```bash
   pip install pytesseract Pillow opencv-python numpy pypdf docx2pdf
   ```

## Usage

Run the main interactive script:
```bash
python main.py
```
You will be presented with a menu:

1. **Extract OCR from Images**:
   - Place your `.jpeg`, `.png`, or other supported image files in the `input/` directory.
   - Extracted data (JSON and Markdown reports) will be saved in the `output/` directory.
2. **Create Combined Batch PDF**:
   - Place your images, `.pdf`, `.doc`, or `.docx` files in the `pdf_input/` directory.
   - Choose the file type from the sub-menu.
   - A single combined PDF will be saved in the `pdf_output/` directory.

## Features

- **Interactive Menu**: Easily choose between OCR and PDF utilities.
- **Batch Processing**: Automatically processes all images in the input folder.
- **Image Preprocessing**: Auto-deskewing, denoising, and contrast enhancement.
- **Structured Extraction**: Extracts opponent and applicant details, opposition grounds, and prayers.
- **Data Export**: Outputs side-by-side comparisons in Markdown and structured JSON.
- **Batch PDF Combiner**: Combines multiple images, PDFs, or Word documents into a single PDF file.

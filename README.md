# Brandex-OCR

Brandex-OCR is a Python-based utility to extract structured data from TM-5 Notice of Opposition forms utilizing Tesseract OCR.

## Prerequisites

1. Install Tesseract OCR on your system.
2. Install Python requirements:
   ```bash
   pip install pytesseract Pillow opencv-python numpy
   ```

## Usage

1. Place your `.jpeg`, `.png`, or other supported image files in the `input/` directory.
2. Run the main processing script:
   ```bash
   python main.py
   ```
3. Extracted data (JSON and Markdown reports) will be saved in the `output/` directory.

## Features

- **Batch Processing**: Automatically processes all images in the input folder.
- **Image Preprocessing**: Auto-deskewing, denoising, and contrast enhancement.
- **Structured Extraction**: Extracts opponent and applicant details, opposition grounds, and prayers.
- **Data Export**: Outputs side-by-side comparisons in Markdown and structured JSON.

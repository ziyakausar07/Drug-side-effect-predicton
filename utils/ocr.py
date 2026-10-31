"""
OCR Module for Drug Label Text Extraction
Uses PIL for image processing with optional Tesseract OCR.
Falls back to a simulated OCR when Tesseract is not installed (academic demo mode).
"""

import os
import re
from difflib import SequenceMatcher

# Try to import pytesseract; if unavailable, use fallback
_TESSERACT_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    _TESSERACT_AVAILABLE = True
except ImportError:
    pass

# Try OpenCV for preprocessing
_CV2_AVAILABLE = False
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    pass


def preprocess_image(image_path):
    """
    Preprocess image for better OCR accuracy using OpenCV.
    Falls back to raw PIL image if OpenCV is not available.
    """
    if _CV2_AVAILABLE:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal with morphological operations
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        processed = cv2.erode(processed, kernel, iterations=1)
        
        # Save preprocessed image
        processed_path = image_path.rsplit('.', 1)[0] + '_processed.png'
        cv2.imwrite(processed_path, processed)
        return processed_path
    
    return image_path


def extract_text_from_image(image_path):
    """
    Extracts text from a drug label/packaging image.
    Uses Tesseract OCR if available, otherwise returns a placeholder for demo purposes.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if _TESSERACT_AVAILABLE:
        try:
            processed_path = preprocess_image(image_path)
            img = Image.open(processed_path)
            
            # Run Tesseract with optimized config for drug labels
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(img, config=custom_config)
            
            # Clean up processed temp file
            if processed_path != image_path and os.path.exists(processed_path):
                os.remove(processed_path)
            
            return text.strip() if text.strip() else "[No text detected in image]"
            
        except Exception as e:
            return f"[OCR Error: {str(e)}]"
    
    # Fallback: academic demo mode — return placeholder
    return "[OCR Demo Mode] Tesseract is not installed. Upload functionality is available but text extraction requires pytesseract + Tesseract-OCR engine. Please install Tesseract-OCR for full functionality."


def match_drug_from_text(extracted_text, drug_list):
    """
    Fuzzy matches extracted OCR text against the known drug catalog.
    
    Args:
        extracted_text: Raw text from OCR
        drug_list: List of drug dicts with 'drug_name', 'generic_name', 'id', 'category'
    
    Returns:
        dict with 'identified', 'drug', 'confidence', 'message'
    """
    if not extracted_text or extracted_text.startswith("["):
        return {
            'identified': False,
            'drug': None,
            'confidence': 0.0,
            'message': 'No readable text was extracted from the image.'
        }
    
    text_lower = extracted_text.lower()
    best_match = None
    best_score = 0.0
    
    for drug in drug_list:
        drug_name = drug.get('drug_name', '')
        generic_name = drug.get('generic_name', '')
        
        # Direct substring match
        if drug_name.lower() in text_lower:
            score = 0.95
        elif generic_name.lower() in text_lower:
            score = 0.90
        else:
            # Fuzzy matching
            name_score = SequenceMatcher(None, drug_name.lower(), text_lower).ratio()
            generic_score = SequenceMatcher(None, generic_name.lower(), text_lower).ratio()
            
            # Check individual words
            words = re.findall(r'[a-zA-Z]{4,}', text_lower)
            word_scores = []
            for word in words:
                ws = max(
                    SequenceMatcher(None, drug_name.lower(), word).ratio(),
                    SequenceMatcher(None, generic_name.lower(), word).ratio()
                )
                word_scores.append(ws)
            
            best_word = max(word_scores) if word_scores else 0
            score = max(name_score, generic_score, best_word)
        
        if score > best_score:
            best_score = score
            best_match = drug
    
    threshold = 0.55
    
    if best_match and best_score >= threshold:
        return {
            'identified': True,
            'drug': best_match,
            'confidence': round(best_score * 100, 1),
            'message': f"Identified '{best_match['drug_name']}' with {round(best_score * 100, 1)}% confidence."
        }
    
    return {
        'identified': False,
        'drug': None,
        'confidence': round(best_score * 100, 1),
        'message': f'Could not reliably identify a drug from the image (best match score: {round(best_score * 100, 1)}%).'
    }

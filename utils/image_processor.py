"""
Image Validation and Processing Utility
Handles upload validation, file sanitization, and image saving.
"""

import os
import uuid
from PIL import Image
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
MAX_DIMENSION = 4096


def allowed_file(filename):
    """Check if the file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_and_save_image(file_obj, upload_folder):
    """
    Validates the uploaded image file, sanitizes the filename,
    and saves it to the upload folder.
    
    Returns:
        tuple: (save_path, filename)
    
    Raises:
        ValueError: If file is invalid or not an image.
    """
    if not file_obj or file_obj.filename == '':
        raise ValueError("No file selected.")
    
    if not allowed_file(file_obj.filename):
        raise ValueError(f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Generate a unique filename
    ext = file_obj.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex[:12]}.{ext}"
    safe_name = secure_filename(unique_name)
    
    save_path = os.path.join(upload_folder, safe_name)
    os.makedirs(upload_folder, exist_ok=True)
    
    # Save temporarily then validate
    file_obj.save(save_path)
    
    try:
        with Image.open(save_path) as img:
            img.verify()
        
        # Re-open (verify closes it) and resize if too large
        with Image.open(save_path) as img:
            w, h = img.size
            if w > MAX_DIMENSION or h > MAX_DIMENSION:
                img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
                img.save(save_path)
                
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise ValueError("The uploaded file is not a valid image.")
    
    return save_path, safe_name

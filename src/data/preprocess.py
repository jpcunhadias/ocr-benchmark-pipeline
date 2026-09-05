import re
import unicodedata

import cv2
import pytesseract
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_orientation_pil(pil_img: Image.Image) -> int:
    """
    Detects the orientation angle of a PIL image using Tesseract's OSD (Orientation and Script Detection).
    Args:
        pil_img (Image.Image): The input PIL image.
    Returns:
        int: The detected orientation angle (0, 90, 180, or 270). Returns 0 if detection fails.
    """
    try:
        osd = pytesseract.image_to_osd(pil_img, lang="osd")
        match = re.search(r"Orientation in degrees: (\d+)", osd)
        if match:
            return int(match.group(1))
    except pytesseract.TesseractError as e:
        logger.warning(f"OSD failed: {e}")
    return 0


def preprocess_image(image_path: str) -> Image.Image:
    """
    Loads an image, applies grayscale conversion, denoising, contrast enhancement,
    adaptive thresholding, and morphological operations to prepare for OCR.
    Also detects and corrects image orientation.
    Args:
        image_path (str): Path to the input image file.
    Returns:
        Image.Image: The preprocessed PIL image, ready for OCR.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(
            f"Could not read image (unsupported format or bad path): {image_path}"
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise using median blur
    denoised = cv2.medianBlur(gray, 3)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    # Binarize image using adaptive thresholding
    binary = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=10,
    )

    # Enhance horizontal lines to help OCR
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    enhanced = cv2.bitwise_or(binary, horizontal_lines)

    # Invert image for OCR
    inverted = cv2.bitwise_not(enhanced)
    pil_img = Image.fromarray(inverted)

    # Detect and correct orientation
    angle = detect_orientation_pil(pil_img)
    if angle in [90, 180, 270]:
        pil_img = pil_img.rotate(-angle, expand=True)

    return pil_img


def preprocess_ocr_text(text: str) -> str:
    """
    Cleans and normalizes OCR-extracted text by removing unwanted characters,
    normalizing Unicode, and filtering out garbage lines.
    """
    # Normalize Unicode
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ASCII", "ignore").decode("ASCII")

    # Fix hyphenated line breaks: wor-\nd -> word
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Remove extra visual noise/symbols
    text = re.sub(r"[|¬•·■✓¤►→●◆▪✓•“”’´`\"'_]+", " ", text)
    text = re.sub(r"[-=]{2,}", " ", text)  # long dashes or equals
    text = re.sub(r"[xX]{5,}", "", text)  # garbage like XXXXXXXX
    text = re.sub(r"[ ]{2,}", " ", text)  # collapse multiple spaces
    text = (
        text.replace(" ,", ",").replace(" .", ".").replace(" :", ":").replace(" ;", ";")
    )

    # Remove broken or irrelevant lines
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()

        # Skip very short lines
        if len(line) < 2:
            continue

        # Skip lines that are only repeated X/K characters (pure OCR noise)
        if re.fullmatch(r"[XxKk\s]{5,}", line):
            continue

        # Skip lines with only common symbols
        if re.fullmatch(r"[0\s\-_/|:.,]+", line):
            continue

        # Skip very short timestamp fragments
        if re.search(r"\d{1,2}[:hH]\d{2}", line) and len(line) < 10:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

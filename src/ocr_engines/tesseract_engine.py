import pytesseract
from pytesseract import Output

from src.data.preprocess import preprocess_image
from src.ocr_engines.base_engine import BaseOCREngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR Engine Implementation.
    """

    def __init__(self, config: dict):
        """
        Initialize Tesseract OCR engine with config.
        Args:
            config (Dict): Configuration parameters.
                Example:
                {
                    "lang": "eng",
                    "oem": 3,
                    "psm": 3
                }
        """
        super().__init__(config)

    def predict(self, image_path: str) -> dict:
        """
        Run OCR on the given image using Tesseract.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: {
                "text": Extracted text (str),
                "confidence": float,
                "engine": "Tesseract"
            }
        """
        img = preprocess_image(image_path)

        # Prepare custom configuration
        tesseract_config = (
            f"--oem {self.config.get('oem', 3)} --psm {self.config.get('psm', 3)}"
        )
        lang = self.config.get("lang", "eng")

        # Perform OCR
        text = pytesseract.image_to_string(img, lang=lang, config=tesseract_config)

        # OCR - extract confidences (assume list of int) and per-word regions
        regions: list[dict] = []
        try:
            data = pytesseract.image_to_data(
                img, lang=lang, config=tesseract_config, output_type=Output.DICT
            )
            confidences = [
                conf for conf in data["conf"] if isinstance(conf, int) and conf >= 0
            ]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            # img is the preprocessed image actually handed to Tesseract --
            # normalizing against its own size (not the source file's) keeps
            # boxes correct even if preprocessing rotated/resized the page.
            img_width, img_height = img.size
            n_words = len(data["text"])
            for i in range(n_words):
                word_text = data["text"][i].strip()
                conf = data["conf"][i]
                if not word_text or not (isinstance(conf, int) and conf >= 0):
                    continue
                regions.append(
                    {
                        "text": word_text,
                        "confidence": float(conf),
                        "bbox": {
                            "left": data["left"][i] / img_width,
                            "top": data["top"][i] / img_height,
                            "width": data["width"][i] / img_width,
                            "height": data["height"][i] / img_height,
                        },
                    }
                )
        except Exception as e:
            logger.warning(f"Confidence estimation failed: {e}")
            avg_conf = 0.0

        return {
            "text": text.strip(),
            "confidence": avg_conf,
            "engine": "Tesseract",
            "regions": regions,
        }


import easyocr
import numpy as np

from src.data.preprocess import preprocess_image
from src.ocr_engines.base_engine import BaseOCREngine


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR Engine Implementation.
    """

    def __init__(self, config: dict):
        """
        Initialize EasyOCR engine with config.
        Args:
            config (Dict): Example:
                {
                    "lang_list": ["pt", "en"],
                    "gpu": false
                }
        """
        super().__init__(config)
        lang_list = self.config.get("lang_list", ["pt"])
        gpu = self.config.get("gpu", False)
        self.reader = easyocr.Reader(lang_list, gpu=gpu)

    def predict(self, image_path: str) -> dict:
        """
        Run OCR on the given image using EasyOCR.
        """
        img = preprocess_image(image_path)
        img_np = np.array(img)

        results = self.reader.readtext(img_np)

        texts = [text for _, text, _ in results]
        confidences = [conf for *_, conf in results if isinstance(conf, float | int)]

        extracted_text = " ".join(texts).strip()
        avg_conf = (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        )

        return {"text": extracted_text, "confidence": avg_conf, "engine": "EasyOCR"}

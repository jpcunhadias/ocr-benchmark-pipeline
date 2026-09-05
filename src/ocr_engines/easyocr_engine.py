import easyocr
import numpy as np

from src.data.preprocess import preprocess_image
from src.ocr_engines.base_engine import BaseOCREngine
from src.ocr_engines.utils import group_regions_into_lines


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

        confidences = [conf for *_, conf in results if isinstance(conf, float | int)]
        avg_conf = (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        )

        # img_np is the preprocessed array actually handed to readtext --
        # normalizing against its own shape (not the source file's) keeps
        # boxes correct even if preprocessing rotated/resized the page.
        img_height, img_width = img_np.shape[:2]
        regions: list[dict] = []
        for quad, text, conf in results:
            xs = [float(point[0]) for point in quad]
            ys = [float(point[1]) for point in quad]
            left, top = min(xs), min(ys)
            width, height = max(xs) - left, max(ys) - top
            regions.append(
                {
                    "text": text,
                    "confidence": (
                        float(conf) if isinstance(conf, float | int) else None
                    ),
                    "bbox": {
                        "left": left / img_width,
                        "top": top / img_height,
                        "width": width / img_width,
                        "height": height / img_height,
                    },
                }
            )

        # readtext() returns detections in no particular line order, so
        # reconstruct reading-order lines from the boxes before joining --
        # a flat " ".join(texts) would otherwise collapse the whole page
        # into a single line, breaking anything (e.g. field_extraction's
        # per-line label lookup) that relies on line structure.
        lines = group_regions_into_lines(regions)
        extracted_text = "\n".join(
            " ".join(region["text"] for region in line) for line in lines
        ).strip()

        return {
            "text": extracted_text,
            "confidence": avg_conf,
            "engine": "EasyOCR",
            "regions": regions,
        }

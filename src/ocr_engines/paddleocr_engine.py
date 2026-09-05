import cv2
import numpy as np

from src.data.preprocess import preprocess_image
from src.ocr_engines.base_engine import BaseOCREngine
from src.ocr_engines.utils import group_regions_into_lines


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR Engine Implementation.
    """

    def __init__(self, config: dict):
        """
        Initialize PaddleOCR engine with config.
        Args:
            config (Dict): Example:
                {
                    "lang": "en",
                    "device": "cpu"
                }
        """
        super().__init__(config)
        from paddleocr import PaddleOCR

        # Document-orientation/unwarping/textline-orientation detection is
        # disabled because preprocess_image() already does grayscale,
        # denoise, contrast, binarize, and orientation correction upstream
        # -- letting PaddleOCR's own doc-preprocessing run too would be
        # redundant and could disagree with the correction already applied.
        self.ocr = PaddleOCR(
            lang=self.config.get("lang", "en"),
            device=self.config.get("device", "cpu"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def predict(self, image_path: str) -> dict:
        """
        Run OCR on the given image using PaddleOCR.
        """
        img = preprocess_image(image_path)
        img_np = np.array(img)
        # preprocess_image() returns a single-channel (grayscale) array --
        # PaddleOCR's detection preprocessing requires 3 channels and
        # raises trying to unpack img.shape otherwise (confirmed live).
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        results = self.ocr.predict(img_np)
        res = results[0] if results else None

        rec_texts = res["rec_texts"] if res else []
        rec_scores = res["rec_scores"] if res else []
        rec_boxes = res["rec_boxes"] if res else []

        confidences = [
            float(score) for score in rec_scores if isinstance(score, float | int)
        ]
        avg_conf = (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        )

        # img_np is the preprocessed array actually handed to predict() --
        # normalizing against its own shape (not the source file's) keeps
        # boxes correct even if preprocessing rotated/resized the page.
        img_height, img_width = img_np.shape[:2]
        regions: list[dict] = []
        for text, score, box in zip(rec_texts, rec_scores, rec_boxes, strict=False):
            left, top, right, bottom = (float(v) for v in box)
            regions.append(
                {
                    "text": text,
                    "confidence": (
                        float(score) if isinstance(score, float | int) else None
                    ),
                    "bbox": {
                        "left": left / img_width,
                        "top": top / img_height,
                        "width": (right - left) / img_width,
                        "height": (bottom - top) / img_height,
                    },
                }
            )

        # PaddleOCR's own detection ordering isn't guaranteed to match
        # reading order, so reconstruct reading-order lines from the boxes
        # before joining -- same reasoning as EasyOCREngine.
        lines = group_regions_into_lines(regions)
        extracted_text = "\n".join(
            " ".join(region["text"] for region in line) for line in lines
        ).strip()

        return {
            "text": extracted_text,
            "confidence": avg_conf,
            "engine": "PaddleOCR",
            "regions": regions,
        }

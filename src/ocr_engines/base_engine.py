from abc import ABC, abstractmethod


class BaseOCREngine(ABC):
    """
    Abstract Base Class for all OCR Engines.
    """

    def __init__(self, config: dict):
        """
        Initialize the OCR engine with a configuration dictionary.
        Args:
            config (Dict): Configuration parameters for the OCR engine.
        """
        self.config = config

    @abstractmethod
    def predict(self, image_path: str) -> dict:
        """
        Perform OCR on the input image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary with keys:
                - 'text': Extracted text (str)
                - 'confidence': Confidence score (float or None)
                - 'engine': Name of the OCR engine (str)
                - 'regions' (optional): Per-detection boxes, list of:
                    {"text": str, "confidence": float | None,
                     "bbox": {"left": float, "top": float,
                              "width": float, "height": float}}
                    bbox values are fractions (0-1) of the image's own
                    width/height (as loaded for OCR, post-preprocessing),
                    not pixels -- keeps boxes comparable across engines and
                    across images rasterized at different resolutions.
        """
        pass

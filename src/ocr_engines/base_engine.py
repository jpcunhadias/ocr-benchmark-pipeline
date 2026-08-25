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
        """
        pass

"""hand_foul — normal / foul hand-position classifier.

Usage:
    from hand_foul import Classifier
    clf = Classifier.from_pretrained("weights/best.pt")
    result = clf.predict("photo.jpg")   # {"label": "foul", "confidence": 0.93, ...}
    clf.show("photo.jpg")                # annotated image window / file
"""

from .data import CLASSES
from .predict import Classifier

__version__ = "0.1.0"
__all__ = ["Classifier", "CLASSES", "__version__"]

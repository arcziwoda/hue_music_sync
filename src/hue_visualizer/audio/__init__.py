"""Audio processing — capture, FFT analysis, beat detection, and section detection."""

from .analyzer import AudioAnalyzer, AudioFeatures, BAND_NAMES
from .beat_detector import BeatDetector, BeatInfo
from .capture import AudioCapture
from .section_detector import Section, SectionDetector, SectionInfo

__all__ = [
    "AudioCapture",
    "AudioAnalyzer",
    "AudioFeatures",
    "BeatDetector",
    "BeatInfo",
    "BAND_NAMES",
    "Section",
    "SectionDetector",
    "SectionInfo",
]

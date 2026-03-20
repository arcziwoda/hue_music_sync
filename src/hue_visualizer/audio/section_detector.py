"""Section detection — drop, buildup, breakdown, and normal classification.

Tracks energy/centroid/onset density trends over sliding windows (8-32 beats)
and classifies the current musical section. Designed to modulate the effect
engine behavior so drops feel like drops and breakdowns feel like breakdowns.

Detection algorithms:
- DROP:      Bass energy spike >3x running average after a period of low bass
- BUILDUP:   Rising RMS + rising centroid + increasing onset density
- BREAKDOWN: Near-zero bass, sustained mid-to-high frequencies
- NORMAL:    Default state, current behavior
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Section(str, Enum):
    """Musical section classification."""

    NORMAL = "normal"
    DROP = "drop"
    BUILDUP = "buildup"
    BREAKDOWN = "breakdown"


@dataclass
class SectionInfo:
    """Section detection output for a single tick."""

    section: Section = Section.NORMAL
    confidence: float = 0.0  # 0-1, how confident we are in this classification
    intensity: float = 0.0  # 0-1, how strong the section effect should be
    beats_in_section: int = 0  # How many beats we've been in this section


class SectionDetector:
    """Detects musical sections (drop, buildup, breakdown) from audio features.

    Maintains sliding windows of bass energy, RMS, spectral centroid,
    and onset count. Classifies the current section based on feature trends
    over configurable beat-aligned windows.

    Typical usage:
        detector = SectionDetector()
        ...
        info = detector.update(bass_energy, rms, centroid, is_beat, bpm)
        # info.section tells you what to do in the effect engine
    """

    def __init__(
        self,
        window_beats: int = 8,
        drop_bass_multiplier: float = 3.0,
        drop_duration_beats: int = 4,
        buildup_min_beats: int = 4,
        breakdown_bass_threshold: float = 0.3,
        breakdown_min_beats: int = 4,
        sample_rate_hz: float = 30.0,
    ):
        """Initialize section detector.

        Args:
            window_beats: Number of beats for the analysis window (default 8).
            drop_bass_multiplier: Bass must exceed running avg by this factor for DROP.
            drop_duration_beats: How many beats a DROP lasts before transitioning.
            buildup_min_beats: Minimum beats of rising trend to trigger BUILDUP.
            breakdown_bass_threshold: Bass must be below this fraction of avg for BREAKDOWN.
            breakdown_min_beats: Minimum beats of low bass to trigger BREAKDOWN.
            sample_rate_hz: How often update() is called (for time-based windows).
        """
        self._window_beats = max(4, window_beats)
        self._drop_bass_multiplier = max(1.5, drop_bass_multiplier)
        self._drop_duration_beats = max(1, drop_duration_beats)
        self._buildup_min_beats = max(2, buildup_min_beats)
        self._breakdown_bass_threshold = max(0.05, min(1.0, breakdown_bass_threshold))
        self._breakdown_min_beats = max(2, breakdown_min_beats)
        self._sample_rate_hz = max(1.0, sample_rate_hz)

        # --- Sliding windows (time-based, converted from beats at runtime) ---
        # We store ~8 seconds of history at the sample rate, enough for
        # any reasonable BPM. The actual beat-aligned analysis uses
        # the BPM to determine how many samples correspond to N beats.
        max_history = int(self._sample_rate_hz * 10)  # 10 seconds of history
        self._bass_history: deque[float] = deque(maxlen=max_history)
        self._rms_history: deque[float] = deque(maxlen=max_history)
        self._centroid_history: deque[float] = deque(maxlen=max_history)
        self._onset_history: deque[bool] = deque(maxlen=max_history)

        # Running average for bass (longer window for baseline comparison)
        # 4 seconds at sample rate — used as "running average" for drop detection
        long_window = int(self._sample_rate_hz * 4)
        self._bass_long_history: deque[float] = deque(maxlen=max(long_window, 30))

        # --- Section state ---
        self._current_section = Section.NORMAL
        self._section_confidence: float = 0.0
        self._section_intensity: float = 0.0
        self._beats_in_section: int = 0
        self._section_start_time: float = 0.0

        # Beat counting within sections
        self._total_beats: int = 0

        # Drop state: track the pre-drop period
        self._pre_drop_low_bass_samples: int = 0

        # Smoothed section intensity for gradual transitions
        self._smoothed_intensity: float = 0.0
        self._intensity_alpha: float = 0.15

    def update(
        self,
        bass_energy: float,
        rms: float,
        centroid: float,
        is_beat: bool,
        bpm: float,
        now: float | None = None,
    ) -> SectionInfo:
        """Process one tick of audio features and classify the current section.

        Call this at the same rate as the effect engine tick (~25-30 Hz).

        Args:
            bass_energy: Current bass energy (0-1 range, sub_bass + bass avg).
            rms: Current RMS energy (0-1 normalized).
            centroid: Spectral centroid in Hz.
            is_beat: Whether a beat was detected this tick.
            bpm: Current estimated BPM (0 if unknown).
            now: Current time (monotonic), defaults to time.monotonic().

        Returns:
            SectionInfo with current classification.
        """
        if now is None:
            now = time.monotonic()

        # --- Append to history buffers ---
        self._bass_history.append(bass_energy)
        self._rms_history.append(rms)
        self._centroid_history.append(centroid)
        self._onset_history.append(is_beat)
        self._bass_long_history.append(bass_energy)

        if is_beat:
            self._total_beats += 1
            self._beats_in_section += 1

        # Need enough history before classifying
        min_samples = int(self._sample_rate_hz * 1.5)
        if len(self._bass_history) < min_samples:
            return SectionInfo()

        # Convert beat window to sample count
        beat_window_samples = self._beats_to_samples(
            self._window_beats, bpm
        )

        # --- Run detection algorithms ---
        # Priority: DROP > BUILDUP > BREAKDOWN > NORMAL
        # (a drop can interrupt anything; buildup leads to drop)

        drop_conf, drop_intensity = self._detect_drop(beat_window_samples)
        buildup_conf, buildup_intensity = self._detect_buildup(beat_window_samples)
        breakdown_conf, breakdown_intensity = self._detect_breakdown(beat_window_samples)

        # --- State machine: transition logic ---
        new_section = self._resolve_section(
            drop_conf, drop_intensity,
            buildup_conf, buildup_intensity,
            breakdown_conf, breakdown_intensity,
            now,
            bpm,
        )

        # Handle section transitions
        if new_section != self._current_section:
            self._current_section = new_section
            self._beats_in_section = 0
            self._section_start_time = now

        # Smooth intensity for gradual transitions
        target_intensity = {
            Section.DROP: drop_intensity,
            Section.BUILDUP: buildup_intensity,
            Section.BREAKDOWN: breakdown_intensity,
            Section.NORMAL: 0.0,
        }[self._current_section]

        self._smoothed_intensity += self._intensity_alpha * (
            target_intensity - self._smoothed_intensity
        )

        # Build output
        return SectionInfo(
            section=self._current_section,
            confidence=self._section_confidence,
            intensity=min(1.0, max(0.0, self._smoothed_intensity)),
            beats_in_section=self._beats_in_section,
        )

    def _beats_to_samples(self, beats: int, bpm: float) -> int:
        """Convert a number of beats to a number of samples at current BPM."""
        if bpm <= 0:
            # No BPM — fall back to ~4 seconds of samples
            return int(self._sample_rate_hz * 4)
        beat_duration_sec = 60.0 / bpm
        total_sec = beats * beat_duration_sec
        return max(10, int(total_sec * self._sample_rate_hz))

    def _detect_drop(self, window_samples: int) -> tuple[float, float]:
        """Detect a drop: bass spike after a period of low bass.

        Returns (confidence, intensity) both in 0-1.
        """
        if len(self._bass_long_history) < 20:
            return 0.0, 0.0

        # Running average bass over the long window (4 seconds)
        bass_long_avg = sum(self._bass_long_history) / len(self._bass_long_history)
        if bass_long_avg < 0.001:
            return 0.0, 0.0

        # Current bass: average over last ~100ms (3-4 samples at 30Hz)
        recent_n = max(1, int(self._sample_rate_hz * 0.1))
        recent_bass_list = list(self._bass_history)[-recent_n:]
        current_bass = sum(recent_bass_list) / len(recent_bass_list)

        # Check the pre-drop condition: was bass below average recently?
        # Look at the 2 seconds BEFORE the most recent 100ms
        pre_drop_n = int(self._sample_rate_hz * 2.0)
        bass_list = list(self._bass_history)
        if len(bass_list) < pre_drop_n + recent_n:
            pre_drop_segment = bass_list[:-recent_n] if recent_n < len(bass_list) else bass_list
        else:
            pre_drop_segment = bass_list[-(pre_drop_n + recent_n):-recent_n]

        if not pre_drop_segment:
            return 0.0, 0.0

        pre_drop_avg = sum(pre_drop_segment) / len(pre_drop_segment)
        pre_drop_was_low = pre_drop_avg < bass_long_avg * 0.8

        # Drop condition: current bass is N times the running average
        bass_ratio = current_bass / (bass_long_avg + 1e-6)
        is_spike = bass_ratio >= self._drop_bass_multiplier

        if is_spike and pre_drop_was_low:
            # Strong drop: bass ratio indicates how dramatic the transition is
            confidence = min(1.0, (bass_ratio - self._drop_bass_multiplier) / 2.0 + 0.7)
            intensity = min(1.0, bass_ratio / (self._drop_bass_multiplier + 2.0))
            return confidence, intensity

        # Weaker drop: just a bass spike without clear pre-drop quiet
        if is_spike:
            confidence = min(1.0, (bass_ratio - self._drop_bass_multiplier) / 3.0 + 0.3)
            intensity = min(1.0, bass_ratio / (self._drop_bass_multiplier + 3.0))
            return confidence, intensity

        return 0.0, 0.0

    def _detect_buildup(self, window_samples: int) -> tuple[float, float]:
        """Detect a buildup: rising RMS + rising centroid + increasing onset density.

        A buildup requires RISING bass/RMS energy. If bass is very low compared
        to the running average, this is a breakdown, not a buildup — even if
        centroid is rising (which is a breakdown characteristic).

        Returns (confidence, intensity) both in 0-1.
        """
        # We need at least a half-window of data to compute trends
        analysis_len = min(window_samples, len(self._rms_history))
        if analysis_len < 10:
            return 0.0, 0.0

        # Reject buildup when bass is very low — that's a breakdown, not a buildup.
        # A buildup should have increasing energy including bass.
        if len(self._bass_long_history) >= 20:
            bass_long_avg = sum(self._bass_long_history) / len(self._bass_long_history)
            if bass_long_avg > 0.01:
                recent_n = max(1, int(self._sample_rate_hz * 0.5))
                recent_bass = list(self._bass_history)[-recent_n:]
                recent_bass_avg = sum(recent_bass) / max(len(recent_bass), 1)
                if recent_bass_avg < bass_long_avg * self._breakdown_bass_threshold:
                    return 0.0, 0.0

        # Get the analysis window
        rms_window = list(self._rms_history)[-analysis_len:]
        centroid_window = list(self._centroid_history)[-analysis_len:]
        onset_window = list(self._onset_history)[-analysis_len:]

        # Compute trends: split window into first half and second half
        half = analysis_len // 2

        rms_first = sum(rms_window[:half]) / max(half, 1)
        rms_second = sum(rms_window[half:]) / max(analysis_len - half, 1)
        rms_rising = rms_second > rms_first * 1.1  # 10% increase

        centroid_first = sum(centroid_window[:half]) / max(half, 1)
        centroid_second = sum(centroid_window[half:]) / max(analysis_len - half, 1)
        centroid_rising = centroid_second > centroid_first * 1.05  # 5% increase

        onset_first = sum(1 for x in onset_window[:half] if x)
        onset_second = sum(1 for x in onset_window[half:] if x)
        onset_rising = onset_second >= onset_first  # At least as many onsets

        # Count how many indicators are positive
        indicators = [rms_rising, centroid_rising, onset_rising]
        positive_count = sum(indicators)

        if positive_count >= 2:
            # At least 2 of 3 indicators rising
            # Confidence based on how strong the trends are
            rms_ratio = (rms_second / max(rms_first, 0.001)) - 1.0  # how much RMS grew
            centroid_ratio = (centroid_second / max(centroid_first, 1.0)) - 1.0

            confidence = min(1.0, 0.3 + positive_count * 0.2 + rms_ratio * 0.5)
            confidence = max(0.0, confidence)

            intensity = min(1.0, rms_ratio * 2.0 + centroid_ratio * 0.5)
            intensity = max(0.0, min(1.0, intensity))

            return confidence, intensity

        return 0.0, 0.0

    def _detect_breakdown(self, window_samples: int) -> tuple[float, float]:
        """Detect a breakdown: low bass energy with sustained mid/high.

        Uses a shorter "recent" window (~1.5 seconds) for the bass check rather
        than the full analysis window, because the analysis window can span the
        transition from normal to breakdown and dilute the signal.

        Returns (confidence, intensity) both in 0-1.
        """
        if len(self._bass_long_history) < 20:
            return 0.0, 0.0

        # Running average bass over the full long window
        bass_long_avg = sum(self._bass_long_history) / len(self._bass_long_history)
        if bass_long_avg < 0.001:
            # If overall bass has always been near zero, can't detect a breakdown
            return 0.0, 0.0

        # Check recent bass: use a SHORT window (~1.5 seconds) so the check
        # isn't diluted by old high-bass samples from before the breakdown.
        recent_len = max(10, int(self._sample_rate_hz * 1.5))
        recent_len = min(recent_len, len(self._bass_history))
        recent_bass = list(self._bass_history)[-recent_len:]
        recent_bass_avg = sum(recent_bass) / len(recent_bass)

        # Bass is low relative to running average
        bass_ratio = recent_bass_avg / (bass_long_avg + 1e-6)
        bass_is_low = bass_ratio < self._breakdown_bass_threshold

        # But we still have some audio activity (not silence)
        recent_rms = list(self._rms_history)[-recent_len:]
        recent_rms_avg = sum(recent_rms) / len(recent_rms)
        has_activity = recent_rms_avg > 0.05  # Not silence

        # Check that the low bass period is sustained (not just a brief dip)
        # Use at least 1 second of data for the sustained check
        sustained_len = max(10, int(self._sample_rate_hz * 1.0))
        sustained_len = min(sustained_len, len(self._bass_history))
        sustained_bass = list(self._bass_history)[-sustained_len:]
        sustained_avg = sum(sustained_bass) / max(len(sustained_bass), 1)
        sustained_low = sustained_avg / (bass_long_avg + 1e-6) < self._breakdown_bass_threshold

        if bass_is_low and has_activity and sustained_low:
            # How dramatically bass has dropped
            drop_ratio = 1.0 - bass_ratio
            confidence = min(1.0, 0.5 + drop_ratio * 0.5)
            intensity = min(1.0, drop_ratio)
            return confidence, intensity

        return 0.0, 0.0

    def _resolve_section(
        self,
        drop_conf: float, drop_intensity: float,
        buildup_conf: float, buildup_intensity: float,
        breakdown_conf: float, breakdown_intensity: float,
        now: float,
        bpm: float,
    ) -> Section:
        """Resolve which section we're in based on detection confidences.

        Implements a simple state machine with hysteresis to prevent rapid
        section switching.
        """
        current = self._current_section

        # --- DROP ---
        # A drop can happen from any state (it's an event, not a sustained state)
        if drop_conf >= 0.5:
            self._section_confidence = drop_conf
            return Section.DROP

        # If we're in DROP, check if it should expire
        if current == Section.DROP:
            # Drops last a fixed number of beats then transition back
            drop_duration_samples = self._beats_to_samples(
                self._drop_duration_beats, bpm
            )
            time_in_section = now - self._section_start_time
            max_duration = drop_duration_samples / max(self._sample_rate_hz, 1.0)
            if time_in_section > max_duration:
                # Drop expired — transition based on what's happening now
                if breakdown_conf >= 0.4:
                    self._section_confidence = breakdown_conf
                    return Section.BREAKDOWN
                self._section_confidence = 0.0
                return Section.NORMAL
            # Still in drop
            self._section_confidence = drop_conf if drop_conf > 0 else self._section_confidence * 0.95
            return Section.DROP

        # --- BREAKDOWN ---
        # Check breakdown BEFORE buildup: low bass with mid/high activity is
        # unambiguous. A false "buildup" with no bass is really a breakdown.
        if breakdown_conf >= 0.5:
            self._section_confidence = breakdown_conf
            return Section.BREAKDOWN

        # If we're in BREAKDOWN and confidence drops, exit
        if current == Section.BREAKDOWN and breakdown_conf < 0.3:
            # Check if we're transitioning to a buildup
            if buildup_conf >= 0.3:
                self._section_confidence = buildup_conf
                return Section.BUILDUP
            self._section_confidence = 0.0
            return Section.NORMAL

        if current == Section.BREAKDOWN:
            self._section_confidence = breakdown_conf
            return Section.BREAKDOWN

        # --- BUILDUP ---
        if buildup_conf >= 0.5:
            self._section_confidence = buildup_conf
            return Section.BUILDUP

        # If we're in BUILDUP and confidence drops, exit to NORMAL
        if current == Section.BUILDUP and buildup_conf < 0.3:
            self._section_confidence = 0.0
            return Section.NORMAL

        if current == Section.BUILDUP:
            # Still building
            self._section_confidence = buildup_conf
            return Section.BUILDUP

        # --- NORMAL ---
        self._section_confidence = 0.0
        return Section.NORMAL

    def reset(self) -> None:
        """Reset all state."""
        self._bass_history.clear()
        self._rms_history.clear()
        self._centroid_history.clear()
        self._onset_history.clear()
        self._bass_long_history.clear()
        self._current_section = Section.NORMAL
        self._section_confidence = 0.0
        self._section_intensity = 0.0
        self._beats_in_section = 0
        self._section_start_time = 0.0
        self._total_beats = 0
        self._pre_drop_low_bass_samples = 0
        self._smoothed_intensity = 0.0

    @property
    def current_section(self) -> Section:
        """Current detected section."""
        return self._current_section

    @property
    def beats_in_section(self) -> int:
        """Number of beats since the current section started."""
        return self._beats_in_section

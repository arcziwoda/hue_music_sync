#!/usr/bin/env python3
"""
Test audio pipeline — capture → FFT → beat detection.
Prints real-time spectrum bars and beat events to terminal.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hue_visualizer.audio import (
    AudioCapture,
    AudioAnalyzer,
    BeatDetector,
    BAND_NAMES,
)


# ANSI colors for terminal
RESET = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
BOLD = "\033[1m"
DIM = "\033[2m"

BAND_COLORS = [RED, RED, YELLOW, GREEN, CYAN, MAGENTA, WHITE]
BAR_WIDTH = 40


def draw_bar(value: float, width: int = BAR_WIDTH, color: str = WHITE) -> str:
    """Draw a horizontal bar for a value 0-1."""
    filled = int(value * width)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"


def main():
    print(f"\n{BOLD}{'=' * 60}")
    print("  Hue Visualizer — Audio Pipeline Test")
    print(f"{'=' * 60}{RESET}\n")

    # List devices
    capture = AudioCapture()
    devices = capture.list_devices()
    print(f"{BOLD}Available input devices:{RESET}")
    for d in devices:
        print(f"  [{d['index']}] {d['name']} (channels={d['channels']}, rate={d['sample_rate']})")

    print(f"\nUsing default device. Press Ctrl+C to stop.\n")

    analyzer = AudioAnalyzer(sample_rate=44100, fft_size=2048, bass_boost=2.0)
    beat_detector = BeatDetector(
        sample_rate=44100, hop_size=1024, threshold_multiplier=1.4, cooldown_ms=300
    )

    beat_count = 0

    with AudioCapture(sample_rate=44100, buffer_size=1024) as capture:
        time.sleep(0.2)  # Let the stream stabilize

        while True:
            frame = capture.wait_for_frame(timeout=0.5)
            if frame is None:
                continue

            # Analyze
            features = analyzer.analyze(frame)
            beat_info = beat_detector.detect(features)

            # Beat indicator
            if beat_info.is_beat:
                beat_count += 1
                beat_marker = f" {BOLD}{RED}♥ BEAT #{beat_count}{RESET}"
            else:
                beat_marker = ""

            # Build display
            lines = ["\033[H\033[J"]  # Clear screen
            lines.append(f"{BOLD}Hue Visualizer — Audio Pipeline Test{RESET}")
            lines.append(f"{DIM}Press Ctrl+C to stop{RESET}\n")

            # Spectrum bars
            lines.append(f"{BOLD}Frequency Bands:{RESET}")
            for i, name in enumerate(BAND_NAMES):
                energy = features.band_energies[i]
                bar = draw_bar(energy, BAR_WIDTH, BAND_COLORS[i])
                lines.append(f"  {name:>12s} {bar} {energy:.3f}")

            lines.append("")

            # Spectral features
            lines.append(f"{BOLD}Spectral Features:{RESET}")
            lines.append(f"  {'RMS':>12s} {draw_bar(min(features.rms * 5, 1.0))} {features.rms:.4f}")
            lines.append(f"  {'Centroid':>12s} {features.spectral_centroid:>8.0f} Hz")
            lines.append(f"  {'Flux':>12s} {features.spectral_flux:>8.2f}")
            lines.append(f"  {'Rolloff':>12s} {features.spectral_rolloff:>8.0f} Hz")
            lines.append(f"  {'Flatness':>12s} {features.spectral_flatness:>8.4f}")

            lines.append("")

            # Beat info
            lines.append(f"{BOLD}Beat Detection:{RESET}")
            bpm_str = f"{beat_info.bpm:.1f}" if beat_info.bpm > 0 else "---"
            conf_str = f"{beat_info.bpm_confidence:.0%}" if beat_info.bpm > 0 else "---"
            lines.append(f"  {'BPM':>12s} {bpm_str}  (confidence: {conf_str})")
            lines.append(f"  {'Beats':>12s} {beat_count}{beat_marker}")
            lines.append(f"  {'Since beat':>12s} {beat_info.time_since_beat:.2f}s")

            print("\n".join(lines), flush=True)

            # Small sleep to keep ~30fps display
            time.sleep(0.016)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{DIM}Stopped.{RESET}")

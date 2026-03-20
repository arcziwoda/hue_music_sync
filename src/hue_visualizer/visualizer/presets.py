"""Genre presets and color palettes.

Each genre has a default palette that is applied when the preset is selected.
Users can still override the palette manually via the palette buttons.
"""

from dataclasses import dataclass


# --- Color Palettes (can be selected independently, but genres have defaults) ---

PALETTES: dict[str, tuple[float, ...]] = {
    # General palettes
    "neon": (280.0, 240.0, 0.0, 320.0),       # deep purple, midnight blue, blood red, magenta
    "warm": (35.0, 15.0, 55.0, 5.0),           # amber, coral, gold, red-orange
    "cool": (200.0, 220.0, 260.0, 180.0),      # ice blue, sky, lavender, teal
    "fire": (0.0, 20.0, 40.0, 350.0),          # red, orange, amber, crimson
    "forest": (120.0, 90.0, 60.0, 150.0),      # green, lime, yellow-green, seafoam
    "ocean": (200.0, 180.0, 240.0, 160.0),     # blue, cyan, violet, aqua
    "sunset": (15.0, 35.0, 300.0, 340.0),      # coral, amber, pink, rose
    "monochrome": (240.0, 230.0, 250.0, 220.0),  # blue shades — subtle variation
    # Genre-specific palettes (from research)
    "techno": (280.0, 240.0, 0.0),             # deep purple, midnight blue, blood red
    "house": (35.0, 15.0, 180.0),              # warm amber, coral, cyan
    "dnb": (120.0, 60.0, 210.0),               # neon green, yellow, electric blue
    "ambient": (200.0, 270.0, 220.0),          # ice blue, lavender, soft sky
}

DEFAULT_PALETTE = "neon"


# --- Genre Presets (beat detection, smoothing, spatial + default palette) ---

@dataclass(frozen=True)
class GenrePreset:
    """Parameter set for a music genre, including default color palette."""

    name: str
    # Beat detection
    beat_threshold: float
    beat_cooldown_ms: float
    bass_boost: float
    # BPM range (octave error protection)
    bpm_min: float
    bpm_max: float
    # Smoothing
    attack_alpha: float
    release_alpha: float
    # Spatial
    spatial_mode: str
    # Flash: exponential decay time constant (seconds). Lower = snappier.
    flash_tau: float
    # Hue drift speed (degrees/second for generative base)
    hue_drift_speed: float
    # Default palette name (key into PALETTES dict)
    default_palette: str = "neon"


PRESETS: dict[str, GenrePreset] = {
    "techno": GenrePreset(
        name="techno",
        beat_threshold=1.3,
        beat_cooldown_ms=280,
        bass_boost=2.5,
        bpm_min=120.0,
        bpm_max=150.0,
        attack_alpha=0.8,
        release_alpha=0.15,
        spatial_mode="frequency_zones",
        flash_tau=0.20,  # Snappy — 200ms decay
        hue_drift_speed=4.0,
        default_palette="techno",  # deep purple, midnight blue, blood red
    ),
    "house": GenrePreset(
        name="house",
        beat_threshold=1.4,
        beat_cooldown_ms=320,
        bass_boost=2.0,
        bpm_min=115.0,
        bpm_max=135.0,
        attack_alpha=0.65,
        release_alpha=0.12,
        spatial_mode="frequency_zones",
        flash_tau=0.30,  # Warm glow — 300ms decay
        hue_drift_speed=8.0,
        default_palette="house",  # warm amber, coral, cyan
    ),
    "dnb": GenrePreset(
        name="dnb",
        beat_threshold=1.2,
        beat_cooldown_ms=200,
        bass_boost=3.0,
        bpm_min=155.0,
        bpm_max=185.0,
        attack_alpha=0.9,
        release_alpha=0.15,
        spatial_mode="wave",
        flash_tau=0.15,  # Very snappy — 150ms for fast beats
        hue_drift_speed=10.0,
        default_palette="dnb",  # neon green, yellow, electric blue
    ),
    "ambient": GenrePreset(
        name="ambient",
        beat_threshold=1.8,
        beat_cooldown_ms=500,
        bass_boost=1.5,
        bpm_min=60.0,
        bpm_max=120.0,
        attack_alpha=0.3,
        release_alpha=0.05,
        spatial_mode="uniform",
        flash_tau=0.50,  # Slow gentle pulse — 500ms
        hue_drift_speed=2.0,
        default_palette="ambient",  # ice blue, lavender, soft sky
    ),
}

DEFAULT_GENRE = "techno"

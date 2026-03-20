# Hue Music Visualizer

Real-time music visualization for Philips Hue lights. Analyzes audio via FFT and beat detection, drives Hue Entertainment API streaming for low-latency light control.

## Architecture

```
[Mic/System Audio] → PyAudio → AudioCapture (thread) → ring buffer
                                    ↓
                        AudioAnalyzer (FFT 2048 Hann) → AudioFeatures
                                    ↓
                        BeatDetector (adaptive threshold + PLL) → BeatInfo
                                    ↓
                    AudioPipeline.process_all() — processes all buffered frames
                                    ↓
                        server/app.py audio_loop (~30 Hz)
                           ↓                    ↓
              WebSocket → Browser        EffectEngine.tick() → list[LightState]
                  (live viz UI)              ↓
                                ColorMapper + SpatialMapper + beat flash + EMA smoothing
                                             ↓
                                EntertainmentController → DTLS → Bridge → Lights (~25 Hz)
```

All audio processing and light control in Python backend. Web UI is a control panel + real-time visualization.

## Current Status

**Sesja 1-5 complete.** Full pipeline: audio → FFT → beat detection (autocorrelation BPM + PLL) → effect engine → Hue Entertainment API + WebSocket → browser visualization. UI controls, light preview, auto-gain, genre presets, stable BPM tracking.

**Code review complete (2026-03-20).** Full audit against research docs — see `BACKLOG.md` for prioritized feature/bug backlog with references to research specs.

## Development

### Package Management — uv only

```bash
uv add package-name          # Add dependency (NEVER edit pyproject.toml manually)
uv add --dev package-name    # Dev dependency
uv sync                      # Install all
uv run python <script>       # Run anything
uv run pytest                # Tests
```

### System Dependencies (macOS)

```bash
brew install portaudio       # PyAudio
brew install mbedtls@2       # Entertainment API (DTLS)
```

### Running

```bash
uv run python -m hue_visualizer                # Main app — server + audio + web UI on localhost:8080
uv run python scripts/test_audio.py            # Test audio pipeline (terminal)
uv run python scripts/test_entertainment.py    # Test Hue Entertainment API
```

## Project Structure

```
src/hue_visualizer/
├── __main__.py                # Entry point (uvicorn + dotenv)
├── core/
│   ├── config.py              # Pydantic Settings (all params from .env)
│   └── exceptions.py          # Custom exception hierarchy
├── bridge/
│   ├── connection.py          # HueBridge REST API wrapper
│   ├── discovery.py           # Bridge discovery & pairing
│   ├── entertainment_controller.py  # Entertainment API (DTLS streaming)
│   └── effects.py             # Tick-based effects (PulseEffect, BreatheEffect, StrobeEffect, FlashDecayEffect, ColorCycleEffect)
├── audio/
│   ├── capture.py             # PyAudio wrapper, threaded capture, ring buffer
│   ├── analyzer.py            # FFT (2048 Hann), 7-band energies, spectral features
│   └── beat_detector.py       # Adaptive beat detection, BPM estimation, PLL
├── visualizer/
│   ├── color_mapper.py        # Audio features → HSV (centroid→hue, RMS→brightness, flatness→saturation)
│   ├── spatial.py             # Per-light distribution (uniform, frequency_zones, wave, mirror)
│   ├── engine.py              # EffectEngine: orchestrates color+spatial+beat+smoothing+safety → LightState[]
│   └── presets.py             # Genre presets (techno, house, DnB, ambient) — parameter sets
├── server/
│   └── app.py                 # FastAPI + WebSocket + AudioPipeline + EffectEngine + EntertainmentController
└── utils/
    └── color_conversion.py    # RGB ↔ HSV ↔ XY (CIE) conversions

frontend/
└── index.html                 # Single-file web UI (dark industrial techno, canvas viz, vanilla JS)

scripts/                       # Setup & test scripts
docs/                          # Research documents
```

## Key Modules

### server/app.py
- `AudioPipeline` — wraps AudioCapture + AudioAnalyzer + BeatDetector, processes all buffered frames per tick (no frame loss for beat detection), latches pending beats
- `ConnectionManager` — WebSocket broadcast to multiple clients
- `audio_loop()` — async background task, ~30 Hz WS broadcast + ~25 Hz light updates via EffectEngine
- Lifespan-managed: auto-starts audio + EffectEngine (always) + optional EntertainmentController, cleans up on shutdown
- Uses Pydantic Settings for all configuration (no more raw os.getenv)
- EffectEngine always active — drives light preview on WebSocket even without bridge
- Bridge connection optional: audio-only mode works without .env bridge vars
- Light preview: sends per-light RGB to browser via `light_preview` field in WS payload
- Control messages from frontend: `set_sensitivity`, `set_bass_boost`, `set_cooldown`, `set_spatial_mode`, `set_genre`, `set_intensity`, `set_effects_size_preset`, `set_color_mode`, `set_palette`
- Genre presets: `_apply_genre_preset()` updates pipeline + engine parameters atomically, uses `set_base_attack_alpha()` to preserve intensity multiplier

### visualizer/ module
- `ColorMapper` — palette-driven hue with spectral centroid as ±20° offset modulator (not primary driver — deliberate deviation from research spec), RMS → brightness (gamma 2.2), flatness → saturation (inverse). All outputs EMA-smoothed
- `SpatialMapper` — 5 modes: uniform, frequency_zones, wave, mirror, chase. NOTE: `distribute()` is dead code — engine reimplements spatial logic internally. Light positions default to linear 0-1 but can be set from bridge entertainment area channel data via `set_positions()` (Task 1.15)
- `EffectEngine` — hybrid reactive-generative orchestrator: ColorMapper → SpatialMapper → beat flash overlay (exponential decay) → per-light asymmetric EMA → safety limiter (max 3Hz flash, no strobe red) → LightState[]. Chase mode: sequential per-bulb activation with beat-synced travel and exponential decay tail. `set_light_positions()` passes bridge positions to spatial mapper + generative layer. Intensity selector (Task 1.12): 3 levels (intense/normal/chill) with multipliers on flash_tau, attack_alpha, max_brightness, beat_threshold — stacks on genre base values. Effects size (Task 1.13): controls how many lights get reactive effects per beat, rotating subset. Light group splitting (Task 1.14): auto-assigns lights to 2-3 groups with palette phase offsets for color diversity
- `effects.py` building blocks (PulseEffect, BreatheEffect, StrobeEffect, FlashDecayEffect, ColorCycleEffect) exist but are NOT integrated with EffectEngine — standalone only

### frontend/index.html
- Dark industrial techno aesthetic (Bebas Neue + IBM Plex Mono, noise grain, scanlines, 1px borders)
- Canvas: spectrum curve (64-bin, bezier smoothed, cyan glow) + 7-band frequency bars
- Browser-side EMA smoothing (attack=0.45, release=0.07) for smooth animation at 60fps
- Beat flash overlay, BPM with confidence opacity, level/beat/centroid meters
- Light preview: colored circles with glow showing per-light RGB state (PREVIEW/STREAMING indicator)
- Controls: sensitivity slider, bass boost slider, cooldown slider
- Spatial mode buttons: UNI / FREQ / WAVE / MIRROR / CHASE
- Genre preset buttons: TECHNO / HOUSE / DNB / AMBIENT
- Intensity buttons: INTENSE / NORMAL / CHILL (Task 1.12)
- Effects size buttons: 1L / 25% / 50% / ALL (Task 1.13)
- WebSocket auto-reconnect with exponential backoff
- Retina DPI canvas scaling

### audio/ pipeline
- `AudioCapture` — threaded PyAudio, ring buffer (deque), `get_all_frames()` for batch processing
- `AudioAnalyzer` — Hann FFT 2048, 7-band energies (auto-gain normalized), bass boost (Fletcher-Munson), spectral centroid/flux/rolloff/flatness. NOTE: no STFT overlap — each 1024-sample buffer is zero-padded to 2048 instead of concatenated with previous frame
- `BeatDetector` — variance-adaptive threshold (uses mean, not median as research recommends), onset function buffer (~4s), BPM via autocorrelation of onset function, PLL with proportional phase+period correction (computes `predicted_next_beat` but engine doesn't use it), octave error protection (configurable BPM range per genre), confidence gating, asymmetric EMA output smoothing. Spectral flux stored in `_flux_history` but never used (dead code)

## Configuration (.env)

**Required for light control (sesja 3+):**
- `BRIDGE_IP` — Hue Bridge IP
- `HUE_USERNAME` — API username (from `scripts/get_clientkey.py`)
- `HUE_CLIENTKEY` — Entertainment API client key
- `ENTERTAINMENT_AREA_ID` — Entertainment area ID (usually "1")

**Optional (with defaults — audio-only mode works without .env):**
- Audio: `SAMPLE_RATE=44100`, `BUFFER_SIZE=1024`, `FFT_SIZE=2048`
- Beat: `BEAT_THRESHOLD_MULTIPLIER=1.4`, `BEAT_COOLDOWN_MS=300`
- Smoothing: `ATTACK_ALPHA=0.7`, `RELEASE_ALPHA=0.1`, `BRIGHTNESS_GAMMA=2.2`, `BASS_BOOST_FACTOR=2.0`
- Safety: `MAX_FLASH_HZ=3.0` (epilepsy limit)
- Server: `SERVER_HOST=0.0.0.0`, `SERVER_PORT=8080`

## Key Technical Constraints

- **Hue Entertainment API**: 25 Hz streaming rate, ~12.5 FPS effective at bulbs
- **End-to-end latency**: 80-120ms typical (audio → light)
- **Safety**: Max 3 Hz flash rate, never strobe saturated red
- **Smoothing**: Asymmetric EMA — fast attack (α=0.5-0.8), slow release (α=0.05-0.15)
- **Beat detection**: Adaptive threshold (variance-based), 300ms min cooldown
- **Brightness**: Gamma-corrected (γ=2.2) for perceptual linearity (Weber-Fechner)
- **Bass**: Boosted 2× to compensate Fletcher-Munson hearing curve

## Known Gaps (from code review)

Key deviations from research specs — details and fixes in `BACKLOG.md`:
- **No STFT overlap** — zero-padding instead of 50% overlap, ~3dB SNR loss (BACKLOG 0.1)
- **25 Hz light send rate** — should be 50-60 Hz for UDP loss compensation (BACKLOG 0.2)
- **No predictive triggering** — PLL computes predicted beats but engine ignores them (BACKLOG 0.4)
- **No hybrid reactive-generative** — purely reactive, boring during quiet passages (BACKLOG 1.1)
- **No multi-layer effects** — monolithic engine, effects.py building blocks unused (BACKLOG 1.2)
- **No section detection** — no drop/buildup/breakdown awareness (BACKLOG 1.3)
- **No per-band onsets** — only bass energy, no kick/snare/hi-hat separation (BACKLOG 1.5)

## Research

Detailed technical research in `docs/`:
- `docs/cemplex_audio_lightning_research.md` — Comprehensive DSP, Hue protocol, perceptual science, architecture
- `docs/ilightshow_research.md` — iLightShow teardown: pre-computed beats, palette system, effects, calibration

## Backlog

See `BACKLOG.md` for the prioritized feature/bug backlog (P0/P1/P2) with references to research document specs.

## Documentation

Always use context7 MCP tools when code generation or library/API docs are needed. Resolve library ID and fetch docs automatically.

## Implementation Plan

See `PLAN.md` for the phased implementation plan with task breakdown.

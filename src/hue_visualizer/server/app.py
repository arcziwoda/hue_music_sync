"""FastAPI server with WebSocket for real-time audio visualization and light control."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..audio import AudioCapture, AudioAnalyzer, AudioFeatures, BeatDetector, BeatInfo, BAND_NAMES, SectionDetector, SectionInfo
from ..core.config import Settings
from ..utils.color_conversion import hsv_to_rgb
from ..visualizer import EffectEngine
from ..visualizer.engine import INTENSITY_LEVELS, INTENSITY_MULTIPLIERS, INTENSITY_NORMAL
from ..visualizer.presets import PRESETS, PALETTES, generate_palette, PALETTE_ALGO_MODES

logger = logging.getLogger(__name__)

SPECTRUM_BINS = 64
WS_RATE_HZ = 30


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WebSocket connected ({len(self.active)} clients)")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"WebSocket disconnected ({len(self.active)} clients)")

    async def broadcast(self, message: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.active:
                self.active.remove(ws)


class AudioPipeline:
    """Audio capture -> FFT analysis -> beat detection pipeline.

    Includes peak-hold buffer (Task 0.5): tracks max RMS, band energies,
    and spectral flux between output ticks so transient peaks are not lost.
    """

    def __init__(self, settings: Settings):
        self.capture = AudioCapture(
            sample_rate=settings.sample_rate,
            buffer_size=settings.buffer_size,
        )
        self.analyzer = AudioAnalyzer(
            sample_rate=settings.sample_rate,
            fft_size=settings.fft_size,
            bass_boost=settings.bass_boost_factor,
        )
        self.beat_detector = BeatDetector(
            sample_rate=settings.sample_rate,
            hop_size=settings.buffer_size,
            threshold_multiplier=settings.beat_threshold_multiplier,
            cooldown_ms=settings.beat_cooldown_ms,
            bpm_min=settings.bpm_min,
            bpm_max=settings.bpm_max,
        )
        self.section_detector = SectionDetector(
            sample_rate_hz=float(settings.sample_rate) / settings.buffer_size,
        )
        self.features = AudioFeatures()
        self.beat_info = BeatInfo()
        self.section_info = SectionInfo()
        self._pending_beat = False
        self._pending_beat_strength = 0.0

        # Per-band onset latching (Task 1.5): latch onsets between output ticks
        self._pending_kick: bool = False
        self._pending_snare: bool = False
        self._pending_hihat: bool = False
        self._peak_kick_energy: float = 0.0
        self._peak_snare_energy: float = 0.0
        self._peak_hihat_energy: float = 0.0

        # Peak-hold buffer: captures transient maxima between output ticks
        self._peak_rms: float = 0.0
        self._peak_band_energies: np.ndarray = np.zeros(7)
        self._peak_spectral_flux: float = 0.0
        self._peak_has_data: bool = False

    def start(self):
        self.capture.start()
        logger.info("Audio pipeline started")

    def stop(self):
        self.capture.stop()
        logger.info("Audio pipeline stopped")

    @property
    def is_running(self) -> bool:
        return self.capture.is_running

    def process_all(self) -> bool:
        """Process all buffered audio frames. Returns True if any were processed."""
        frames = self.capture.get_all_frames()
        for frame in frames:
            self.features = self.analyzer.analyze(frame)
            self.beat_info = self.beat_detector.detect(self.features)
            if self.beat_info.is_beat:
                self._pending_beat = True
                self._pending_beat_strength = max(
                    self._pending_beat_strength, self.beat_info.beat_strength
                )

            # Latch per-band onsets (Task 1.5)
            if self.beat_info.kick_onset:
                self._pending_kick = True
                self._peak_kick_energy = max(
                    self._peak_kick_energy, self.beat_info.kick_energy
                )
            if self.beat_info.snare_onset:
                self._pending_snare = True
                self._peak_snare_energy = max(
                    self._peak_snare_energy, self.beat_info.snare_energy
                )
            if self.beat_info.hihat_onset:
                self._pending_hihat = True
                self._peak_hihat_energy = max(
                    self._peak_hihat_energy, self.beat_info.hihat_energy
                )

            # Section detection: runs on every frame alongside beat detector
            self.section_info = self.section_detector.update(
                bass_energy=self.features.bass_energy,
                rms=self.features.rms,
                centroid=self.features.spectral_centroid,
                is_beat=self.beat_info.is_beat,
                bpm=self.beat_info.bpm,
            )

            # Update peak-hold buffer with per-frame maxima
            self._peak_rms = max(self._peak_rms, self.features.rms)
            self._peak_band_energies = np.maximum(
                self._peak_band_energies, self.features.band_energies
            )
            self._peak_spectral_flux = max(
                self._peak_spectral_flux, self.features.spectral_flux
            )
            self._peak_has_data = True

        return len(frames) > 0

    def consume_beat(self) -> tuple[bool, float]:
        """Return and clear pending beat flag."""
        had_beat = self._pending_beat
        strength = self._pending_beat_strength
        self._pending_beat = False
        self._pending_beat_strength = 0.0
        return had_beat, strength

    def consume_band_onsets(self) -> tuple[bool, bool, bool, float, float, float]:
        """Return and clear pending per-band onset flags (Task 1.5).

        Returns:
            (kick_onset, snare_onset, hihat_onset,
             kick_energy, snare_energy, hihat_energy)
        """
        kick = self._pending_kick
        snare = self._pending_snare
        hihat = self._pending_hihat
        kick_e = self._peak_kick_energy
        snare_e = self._peak_snare_energy
        hihat_e = self._peak_hihat_energy

        self._pending_kick = False
        self._pending_snare = False
        self._pending_hihat = False
        self._peak_kick_energy = 0.0
        self._peak_snare_energy = 0.0
        self._peak_hihat_energy = 0.0

        return kick, snare, hihat, kick_e, snare_e, hihat_e

    def consume_features(self) -> AudioFeatures:
        """Return peak-held features and reset the peak buffer.

        Returns features with the maximum RMS, band energies, and spectral
        flux observed since the last consume call, preserving the most recent
        spectrum and other scalar features for display.
        """
        if not self._peak_has_data:
            return self.features

        # Build features with peak values overlaid on the latest frame
        peak_features = AudioFeatures(
            band_energies=self._peak_band_energies.copy(),
            spectral_centroid=self.features.spectral_centroid,
            spectral_flux=self._peak_spectral_flux,
            spectral_rolloff=self.features.spectral_rolloff,
            spectral_flatness=self.features.spectral_flatness,
            rms=self._peak_rms,
            peak=self.features.peak,
            spectrum=self.features.spectrum,
        )

        # Reset peak buffer
        self._peak_rms = 0.0
        self._peak_band_energies = np.zeros(7)
        self._peak_spectral_flux = 0.0
        self._peak_has_data = False

        return peak_features


def _prepare_spectrum(spectrum_db: np.ndarray, n_bins: int = SPECTRUM_BINS) -> list[float]:
    """Downsample FFT spectrum to n_bins, normalized to 0-1."""
    if len(spectrum_db) == 0:
        return [0.0] * n_bins

    # Normalize dB to 0-1 range
    min_db, max_db = -80.0, 0.0
    normalized = np.clip((spectrum_db - min_db) / (max_db - min_db), 0, 1)

    # Log-spaced bin edges for better low-frequency resolution
    n_fft = len(normalized)
    edges = np.unique(
        np.logspace(0, np.log10(max(n_fft, 2)), n_bins + 1, dtype=int).clip(0, n_fft - 1)
    )

    result = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        result.append(float(np.max(normalized[s : max(e, s + 1)])))

    # Pad or trim to exact n_bins
    while len(result) < n_bins:
        result.append(0.0)
    return result[:n_bins]


def _light_states_to_preview(engine: EffectEngine) -> list[dict]:
    """Convert engine's current per-light smoothed state to RGB dicts for UI preview."""
    preview = []
    for light in engine._lights:
        r, g, b = hsv_to_rgb(light.hue, light.saturation, light.brightness)
        preview.append({"r": r, "g": g, "b": b})
    return preview


# --- Global state ---
manager = ConnectionManager()
pipeline: AudioPipeline | None = None
effect_engine: EffectEngine | None = None
entertainment_ctrl = None  # EntertainmentController | None
settings: Settings | None = None
current_genre: str = "techno"
current_palette: str = "neon"
current_intensity: str = INTENSITY_NORMAL


async def audio_loop():
    """Background task: process audio, drive lights, broadcast to WebSocket clients.

    Uses target-based timing (Task 0.7) to prevent drift from processing time.
    Light send rate uses settings.fps_target (Task 0.2) for UDP oversampling.
    """
    ws_interval = 1.0 / WS_RATE_HZ
    # Task 0.2: Use fps_target from config (default 50 Hz) instead of hardcoded 25 Hz.
    # Oversampling at 50-60 Hz compensates for UDP packet loss; bridge decimates internally.
    light_interval = 1.0 / settings.fps_target if settings else 1.0 / 50
    last_light_tick = time.monotonic()

    while True:
        # Task 0.7: Target-based timing — compute next tick before processing
        next_ws_tick = time.monotonic() + ws_interval

        now = time.monotonic()

        if pipeline and pipeline.is_running:
            had_frames = pipeline.process_all()
            had_beat, beat_strength = pipeline.consume_beat()
            kick, snare, hihat, kick_e, snare_e, hihat_e = pipeline.consume_band_onsets()

            # Task 0.5: Consume peak-held features for this output tick
            output_features = pipeline.consume_features()

            # --- Effect engine tick (always, for preview + optional light output) ---
            dt_light = now - last_light_tick
            if effect_engine and dt_light >= light_interval:
                last_light_tick = now

                beat_for_engine = BeatInfo(
                    is_beat=had_beat,
                    bpm=pipeline.beat_info.bpm,
                    bpm_confidence=pipeline.beat_info.bpm_confidence,
                    beat_strength=beat_strength,
                    predicted_next_beat=pipeline.beat_info.predicted_next_beat,
                    time_since_beat=pipeline.beat_info.time_since_beat,
                    kick_onset=kick,
                    snare_onset=snare,
                    hihat_onset=hihat,
                    kick_energy=kick_e,
                    snare_energy=snare_e,
                    hihat_energy=hihat_e,
                )

                try:
                    light_states = effect_engine.tick(
                        output_features, beat_for_engine, dt_light,
                        section_info=pipeline.section_info,
                    )
                    # Task B.5: Send all light states as a batch instead of one-by-one
                    if entertainment_ctrl:
                        entertainment_ctrl.set_light_states_batch(light_states)
                except Exception as e:
                    logger.error(f"Light control error: {e}")

            # --- WebSocket broadcast ---
            if had_frames and manager.active:
                f = output_features
                b = pipeline.beat_info

                data = {
                    "type": "audio",
                    "spectrum": _prepare_spectrum(f.spectrum),
                    "bands": [round(v, 4) for v in f.band_energies.tolist()],
                    "band_names": BAND_NAMES,
                    "beat": {
                        "is_beat": had_beat,
                        "bpm": round(b.bpm, 1),
                        "confidence": round(b.bpm_confidence, 2),
                        "strength": round(beat_strength, 2),
                    },
                    "rms": round(f.rms, 4),
                    "peak": round(f.peak, 4),
                    "spectral_centroid": round(f.spectral_centroid, 1),
                    "spectral_flatness": round(f.spectral_flatness, 3),
                }

                if effect_engine:
                    data["lights_active"] = entertainment_ctrl is not None
                    data["spatial_mode"] = effect_engine.spatial_mapper.mode
                    data["light_preview"] = _light_states_to_preview(effect_engine)
                    data["genre"] = current_genre
                    data["palette"] = current_palette
                    data["color_mode"] = effect_engine.color_mapper.color_mode
                    data["reactive_weight"] = round(effect_engine.reactive_weight, 3)
                    data["energy_level"] = round(effect_engine.energy_level, 3)
                    data["section"] = {
                        "name": effect_engine.current_section.value,
                        "intensity": round(effect_engine.section_intensity, 3),
                    }
                    # Task 1.12: Intensity selector state
                    data["intensity_level"] = effect_engine.intensity_level
                    # Task 1.13: Effects size state
                    data["effects_size"] = round(effect_engine.effects_size, 2)
                    # Task 1.14: Light group info
                    data["light_groups"] = effect_engine.light_groups
                    # Task 2.5: Safe mode state
                    data["safe_mode"] = effect_engine.safe_mode
                    # Task 2.19: Saturation boost state
                    data["saturation_boost"] = round(effect_engine.saturation_boost, 2)
                    # Task 2.16: White flash mode state
                    data["white_flash"] = effect_engine.white_flash_mode
                    # Task 2.6: Calibration delay state
                    data["calibration_delay"] = round(effect_engine.calibration_delay_ms)
                    # Task 2.8: Brightness min/max state
                    data["brightness_min"] = round(effect_engine.brightness_min, 2)
                    data["brightness_max"] = round(effect_engine.brightness_max, 2)

                # Task B.10: Include control state so frontend can sync on (re)connect
                if pipeline:
                    data["sensitivity"] = round(pipeline.beat_detector.base_threshold, 1)
                    data["bass_boost"] = round(pipeline.analyzer.bass_boost, 1)
                    data["cooldown_ms"] = round(pipeline.beat_detector.cooldown_sec * 1000)

                await manager.broadcast(json.dumps(data))

        # Task 0.7: Sleep only the remaining time until next tick
        sleep_time = next_ws_tick - time.monotonic()
        await asyncio.sleep(max(0, sleep_time))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop audio pipeline, effect engine, and entertainment controller."""
    global pipeline, effect_engine, entertainment_ctrl, settings

    settings = Settings()

    pipeline = AudioPipeline(settings)

    try:
        pipeline.start()
    except Exception as e:
        logger.error(f"Failed to start audio capture: {e}")
        logger.warning("Running without audio -- connect a microphone and restart")

    # --- Effect engine (always created — drives UI preview even without bridge) ---
    num_lights = settings.num_lights

    # --- Entertainment API (optional — only if bridge configured) ---
    if settings.bridge_ip and settings.hue_username and settings.hue_clientkey:
        try:
            from ..bridge.entertainment_controller import EntertainmentController

            ctrl = EntertainmentController(
                bridge_ip=settings.bridge_ip,
                username=settings.hue_username,
                clientkey=settings.hue_clientkey,
                entertainment_area_id=settings.entertainment_area_id,
            )
            ctrl.connect()
            entertainment_ctrl = ctrl
            num_lights = ctrl._num_lights or settings.num_lights
            logger.info(f"Bridge connected: {num_lights} lights")
        except Exception as e:
            logger.error(f"Failed to start entertainment API: {e}")
            logger.warning("Running without light control")
            entertainment_ctrl = None
    else:
        logger.info("No bridge configured -- audio + preview mode")

    effect_engine = EffectEngine(
        num_lights=num_lights,
        gamma=settings.brightness_gamma,
        attack_alpha=settings.attack_alpha,
        release_alpha=settings.release_alpha,
        max_flash_hz=settings.max_flash_hz,
        spatial_mode=settings.spatial_mode,
        latency_compensation_ms=settings.latency_compensation_ms,
        predictive_confidence_threshold=settings.predictive_confidence_threshold,
        generative_hue_cycle_period=settings.generative_hue_cycle_period,
        generative_breathing_rate_hz=settings.generative_breathing_rate_hz,
        generative_breathing_min=settings.generative_breathing_min,
        generative_breathing_max=settings.generative_breathing_max,
    )

    # Task 2.6: Apply calibration delay from config
    if settings.calibration_delay_ms > 0:
        effect_engine.set_calibration_delay(settings.calibration_delay_ms)
    # Task 2.8: Apply brightness min/max from config
    if settings.brightness_min > 0:
        effect_engine.set_brightness_min(settings.brightness_min)
    if settings.brightness_max < 1.0:
        effect_engine.set_brightness_max(settings.brightness_max)

    # Task 1.15: Pass bridge light positions to spatial mapper when available
    if entertainment_ctrl and entertainment_ctrl.light_positions:
        effect_engine.set_light_positions(entertainment_ctrl.light_positions)
        logger.info(
            f"Light positions from bridge applied: "
            f"{[round(p, 3) for p in entertainment_ctrl.light_positions]}"
        )

    logger.info(
        f"Effect engine: {num_lights} lights, mode={settings.spatial_mode}, "
        f"latency_comp={settings.latency_compensation_ms}ms, "
        f"predictive_thresh={settings.predictive_confidence_threshold}, "
        f"generative_breathing={settings.generative_breathing_rate_hz}Hz"
    )

    # Task 0.6: Apply genre preset at startup so engine params match the preset
    _apply_genre_preset(current_genre)

    task = asyncio.create_task(audio_loop())
    logger.info(f"Server ready -- open http://localhost:{settings.server_port}")

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    if pipeline:
        pipeline.stop()
    if entertainment_ctrl:
        try:
            entertainment_ctrl.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting entertainment API: {e}")


app = FastAPI(title="Hue Visualizer", lifespan=lifespan)


@app.get("/")
async def index():
    """Serve the main visualization UI."""
    html_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                _handle_control(msg)
            except (json.JSONDecodeError, ValueError):
                pass
    except WebSocketDisconnect:
        manager.disconnect(ws)


def _handle_control(msg: dict):
    """Apply control messages from the frontend."""
    if not pipeline:
        return

    t = msg.get("type")

    if t == "set_sensitivity":
        v = float(msg["value"])
        pipeline.beat_detector.set_threshold(v)
        logger.info(f"Beat threshold -> {v}")

    elif t == "set_bass_boost":
        v = float(msg["value"])
        pipeline.analyzer.bass_boost = v
        logger.info(f"Bass boost -> {v}")

    elif t == "set_cooldown":
        v = float(msg["value"])
        pipeline.beat_detector.cooldown_sec = v / 1000.0
        pipeline.beat_detector.auto_cooldown = False  # Manual override
        logger.info(f"Beat cooldown -> {v}ms (auto off)")

    elif t == "set_spatial_mode" and effect_engine:
        mode = msg.get("value", "frequency_zones")
        effect_engine.set_spatial_mode(mode)
        logger.info(f"Spatial mode -> {mode}")

    elif t == "set_genre":
        global current_genre
        genre = msg.get("value", "techno")
        current_genre = genre
        _apply_genre_preset(genre)

    elif t == "set_color_mode" and effect_engine:
        mode = msg.get("value", "palette")
        effect_engine.set_color_mode(mode)
        logger.info(f"Color mode -> {mode}")

    elif t == "set_palette":
        global current_palette
        name = msg.get("value", "neon")
        palette = PALETTES.get(name)
        if palette and effect_engine:
            current_palette = name
            effect_engine.set_palette(palette)
            logger.info(f"Palette -> {name}")

    elif t == "set_intensity":
        global current_intensity
        level = msg.get("value", INTENSITY_NORMAL)
        if level in INTENSITY_LEVELS and effect_engine:
            current_intensity = level
            effect_engine.set_intensity(level)
            # Apply beat threshold multiplier to pipeline
            if pipeline:
                base_thresh = pipeline.beat_detector.base_threshold
                mult = effect_engine.get_intensity_beat_threshold_multiplier()
                pipeline.beat_detector.set_threshold(base_thresh * mult)
            logger.info(f"Intensity -> {level}")

    elif t == "set_effects_size" and effect_engine:
        v = float(msg["value"])
        effect_engine.set_effects_size(v)
        logger.info(f"Effects size -> {v}")

    elif t == "set_safe_mode" and effect_engine:
        enabled = bool(msg.get("value", False))
        effect_engine.set_safe_mode(enabled)
        logger.info(f"Safe mode -> {'ON' if enabled else 'OFF'}")

    elif t == "set_effects_size_preset" and effect_engine:
        # Convenience: accepts named presets "1L", "25%", "50%", "ALL"
        preset = msg.get("value", "ALL")
        if preset == "1L":
            size = 1.0 / max(effect_engine.num_lights, 1)
        elif preset == "25%":
            size = 0.25
        elif preset == "50%":
            size = 0.5
        else:
            size = 1.0
        effect_engine.set_effects_size(size)
        logger.info(f"Effects size preset -> {preset} ({size})")

    elif t == "set_palette_algo" and effect_engine:
        # Task 2.10: Algorithmic palette generation
        algo_mode = msg.get("mode", "complementary")
        base_hue = float(msg.get("base_hue", 200))
        if algo_mode in PALETTE_ALGO_MODES:
            palette = generate_palette(algo_mode, base_hue)
            current_palette = f"algo:{algo_mode}"
            effect_engine.set_palette(palette)
            logger.info(
                f"Algorithmic palette -> {algo_mode} base={base_hue:.0f} "
                f"hues={[round(h, 1) for h in palette]}"
            )

    elif t == "set_saturation" and effect_engine:
        # Task 2.19: Saturation slider
        v = float(msg.get("value", 1.0))
        effect_engine.set_saturation_boost(v)
        logger.info(f"Saturation boost -> {v:.2f}")

    elif t == "set_white_flash" and effect_engine:
        # Task 2.16: White flash mode toggle
        enabled = bool(msg.get("value", False))
        effect_engine.set_white_flash_mode(enabled)
        logger.info(f"White flash mode -> {'ON' if enabled else 'OFF'}")

    elif t == "trigger_flash" and effect_engine:
        # Task 2.17: Manual single flash
        effect_engine.trigger_manual_flash()
        logger.info("Manual flash triggered")

    elif t == "trigger_strobe" and effect_engine:
        # Task 2.17: Manual strobe burst (3 flashes)
        effect_engine.trigger_manual_strobe()
        logger.info("Manual strobe triggered")

    elif t == "set_calibration_delay" and effect_engine:
        # Task 2.6: Manual calibration delay
        v = float(msg.get("value", 0))
        effect_engine.set_calibration_delay(v)
        logger.info(f"Calibration delay -> {v}ms (effective: {effect_engine.effective_latency_compensation_ms:.0f}ms)")

    elif t == "set_brightness_min" and effect_engine:
        # Task 2.8: Brightness floor
        v = float(msg.get("value", 0))
        effect_engine.set_brightness_min(v)
        logger.info(f"Brightness min -> {v:.2f}")

    elif t == "set_brightness_max" and effect_engine:
        # Task 2.8: Brightness cap
        v = float(msg.get("value", 1.0))
        effect_engine.set_brightness_max(v)
        logger.info(f"Brightness max -> {v:.2f}")


def _apply_genre_preset(genre: str) -> None:
    """Apply a genre preset to pipeline + effect engine, including default palette.

    Task B.4: Uses public setter methods instead of accessing private attributes.
    Task 1.11: Genre preset now includes default_palette, applied atomically.
    """
    global current_palette

    preset = PRESETS.get(genre)
    if not preset:
        return

    if pipeline:
        pipeline.beat_detector.set_threshold(preset.beat_threshold)
        pipeline.beat_detector.set_cooldown(preset.beat_cooldown_ms)
        pipeline.beat_detector.set_bpm_range(preset.bpm_min, preset.bpm_max)
        pipeline.beat_detector.reset()
        pipeline.analyzer.bass_boost = preset.bass_boost

    if effect_engine:
        # Task 1.12: Use set_base_attack_alpha so intensity multiplier is preserved
        effect_engine.set_base_attack_alpha(preset.attack_alpha)
        effect_engine.release_alpha = preset.release_alpha
        effect_engine.set_spatial_mode(preset.spatial_mode)
        effect_engine.set_flash_tau(preset.flash_tau)
        effect_engine.set_hue_drift_speed(preset.hue_drift_speed)

        # Apply genre's default palette (Task 1.11)
        palette = PALETTES.get(preset.default_palette)
        if palette:
            current_palette = preset.default_palette
            effect_engine.set_palette(palette)

    logger.info(f"Genre preset -> {genre} (palette: {preset.default_palette})")

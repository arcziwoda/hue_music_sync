# Hue Visualizer — Backlog

Feature backlog generated from code review against research documents:
- `docs/cemplex_audio_lightning_research.md` (dalej: **RESEARCH**)
- `docs/ilightshow_research.md` (dalej: **ILS**)

Priorytet: **P0** = krytyczny impact na jakość, **P1** = ważny feature gap, **P2** = nice-to-have / polish.

---

## P0 — Krytyczne poprawki jakości

### 0.1 STFT 50% overlap
**Problem:** Każdy 1024-sample buffer jest zero-paddowany do 2048 zamiast łączony z poprzednim. Hann window działa na połowie zer — utrata ~3dB SNR, spectral smearing wpływa na centroid i band energies.
**Plik:** `src/hue_visualizer/audio/analyzer.py:128-133`
**Fix:** Dodaj `self._prev_frame` buffer, concatenate z bieżącym frame → pełne 2048 samples pod Hann.
**Ref:** RESEARCH §1 "Audio arrives in discrete chunks" — "frames typically overlap by 50-75%. With a 2,048-sample analysis window and a hop size of 1,024, each frame shares half its data with the adjacent frame."

### 0.2 Light send rate 50-60 Hz (oversampling)
**Problem:** Hardcoded `LIGHT_RATE_HZ = 25` w `app.py:24`. Research mówi: oversample bo UDP gubi pakiety, bridge decimuje do 25 Hz. Config `fps_target=50` istnieje ale jest martwy kod.
**Plik:** `src/hue_visualizer/server/app.py:24`, `src/hue_visualizer/core/config.py:73-78`
**Fix:** Użyj `settings.fps_target` zamiast hardcoded 25. Domyślnie 50 Hz.
**Ref:** RESEARCH §9 "Philips recommends sending at 50-60 Hz to compensate for UDP packet loss — the bridge decimates to 25 Hz internally." ILS §Latency "50-60 Hz recommended (intentional oversampling to compensate for UDP packet loss)."

### 0.3 Moving median zamiast mean w beat detection
**Problem:** `np.mean(history)` zamiast `np.median(history)` — outlier spike zawyża średnią na wiele klatek, powodując missed beats.
**Plik:** `src/hue_visualizer/audio/beat_detector.py:126`
**Fix:** Jednolinijkowa zmiana: `np.mean` → `np.median`.
**Ref:** RESEARCH §3 "Moving median is preferred over moving average because it resists inflation from outlier peaks."

### 0.4 Predictive beat triggering
**Problem:** PLL oblicza `predicted_next_beat` ale EffectEngine reaguje tylko na przeszłe beaty. Brak kompensacji latencji 80-120ms — światła zawsze spóźnione.
**Pliki:** `src/hue_visualizer/audio/beat_detector.py:259-260` (oblicza), `src/hue_visualizer/visualizer/engine.py:117` (ignoruje)
**Fix:** Engine powinien wysyłać komendy ~80ms przed predicted beat. Wymaga confidence gating (>80% locked).
**Ref:** RESEARCH §4 "lights can be triggered slightly before the beat arrives, compensating for system latency. next_beat = last_beat_time + beat_period." ILS §Pre-computed beats "predictive beat triggering is the critical latency-compensation mechanism... commands sent before the beat actually plays from speakers."

### 0.5 Peak-hold dla non-beat audio features
**Problem:** `pipeline.features` nadpisywany co klatkę w `process_all()`. Między tickami output (25-30 Hz) transienty RMS/band energy mogą być zgubione.
**Plik:** `src/hue_visualizer/server/app.py:96-107`
**Fix:** Dodaj peak-hold buffer: trackuj max RMS, max band energies, max flux między output ticks. Reset po consume.
**Ref:** RESEARCH §9 "maintain a 'current target state' buffer... Use peak-hold logic to avoid losing transient peaks between sends." §11 "decouple analysis from output rate... peak-hold logic to preserve transient impacts."

### 0.6 Genre preset nie aplikowany na startup
**Problem:** `current_genre = "techno"` ale `_apply_genre_preset("techno")` nigdy nie wywołane przy starcie. Engine defaults ≠ techno preset values (np. `flash_tau` 0.25 vs 0.20).
**Plik:** `src/hue_visualizer/server/app.py:275-282`
**Fix:** Wywołaj `_apply_genre_preset(current_genre)` po utworzeniu pipeline i engine w lifespan.

### 0.7 Audio loop timing drift
**Problem:** `await asyncio.sleep(ws_interval)` — fixed sleep po processing. Rzeczywisty okres = interval + processing_time → dryfuje poniżej 30 Hz.
**Plik:** `src/hue_visualizer/server/app.py:232`
**Fix:** Target-based sleep: `next_tick = now + interval; ... await asyncio.sleep(max(0, next_tick - time.monotonic()))`.

---

## P1 — Ważne feature gaps

### 1.1 Hybrid reactive-generative model
**Problem:** System jest czysto reaktywny. W cichych pasażach światła prawie nie reagują. Research opisuje generative base layer który zawsze daje ładne światło.
**Plik:** `src/hue_visualizer/visualizer/engine.py`
**Fix:** Dodaj generative base: wolna rotacja hue (30-60s cykl), breathing (0.25 Hz), spatial waves. Blend z reactive: quiet → 80% generative, energetic → 80% reactive.
**Ref:** RESEARCH §7 "Pure reactive lighting produces tight sync but boring lights during quiet passages... The optimal strategy is hybrid: a generative base layer (slow hue rotation completing one cycle every 30-60 seconds, gentle breathing at 0.25 Hz, gradual spatial waves) modulated by reactive triggers. During quiet passages, the generative base dominates (~80%); during energetic sections, reactive effects take over (~80%)."

### 1.2 Multi-layer effect system z blend modes
**Problem:** Engine jest monolityczny — jeden przejazd. `effects.py` ma gotowe building blocks (Pulse, Breathe, Strobe, ColorCycle, FlashDecay) ale żaden nie jest zintegrowany.
**Pliki:** `src/hue_visualizer/visualizer/engine.py`, `src/hue_visualizer/bridge/effects.py`
**Fix:** Layer system: base layer (generative) + reactive layer (beat/bass) + section layer (intensity). Blend modes: maximum (beat flashes on base), multiplicative (brightness envelopes), additive (independent colors).
**Ref:** RESEARCH §11 "The effect engine should support multiple simultaneous layers with blending. A base layer provides continuous generative patterns. A reactive layer overlays beat-triggered brightness pulses. A section-detection layer modulates overall intensity and palette. Blending modes: maximum, multiplicative, additive."

### 1.3 Section detection (drop / buildup / breakdown)
**Problem:** Brak śledzenia trendów energii. Światła nie reagują na strukturę utworu — drop wygląda tak samo jak intro.
**Plik:** `src/hue_visualizer/audio/beat_detector.py` lub nowy moduł
**Fix:** Śledź energy/centroid/onset density w oknie 8-32 beatów:
- **Drop**: bass spike >3× running avg po okresie niskiego basu → flash all max + saturated warm
- **Buildup**: rising RMS + rising centroid + increasing onset density → ramp brightness, shift warm
- **Breakdown**: near-zero bass, sustained mid-to-high → dim to 20-40%, cool pastels, breathing
**Ref:** RESEARCH §7 "By tracking energy, centroid, onset rate, and spectral flux over 8-32 beat windows, the system distinguishes sections automatically." ILS §Effect modes "Strobe at Drops uses pre-computed section analysis to identify buildups and drops." RESEARCH §4 "Drops produce a sudden, massive bass energy spike (>3× running average) following a period of reduced bass."

### 1.4 Spectral centroid jako primary hue driver (opcja)
**Problem:** Centroid zdegradowany do ±20° offset. Research traktuje centroid jako główne mapowanie hue.
**Plik:** `src/hue_visualizer/visualizer/color_mapper.py:47-54`
**Fix:** Dodaj tryb "centroid-driven" obok "palette-driven" — przełączalny z UI. Log-scaled 100Hz→red do 10kHz→violet (hue 0-300°).
**Ref:** RESEARCH §6 "spectral centroid can directly drive hue as a single continuous value, naturally shifting warmer as bass dominates and cooler as treble dominates. Map the log-scaled centroid to hue 0-300°."

### 1.5 Per-band onset detection
**Problem:** Tylko łączny bass_energy do beat detection. Brak rozróżnienia kick/snare/hi-hat.
**Plik:** `src/hue_visualizer/audio/beat_detector.py`
**Fix:** Osobne onset functions per band: low (kicks) → bass pulse, mid (snares) → white flash, high (hi-hats) → sparkle na random bulbach.
**Ref:** RESEARCH §3 "perform onset detection separately per frequency band: low-band onsets (kicks) trigger bass pulses, mid-band onsets (snares) trigger white flashes, high-band onsets (hi-hats) trigger sparkle effects on random bulbs."

### 1.6 Spectral flux onset detection
**Problem:** Flux jest obliczany i stored w `_flux_history` ale nigdy nie użyty. Dead code.
**Plik:** `src/hue_visualizer/audio/beat_detector.py:64,140-141`
**Fix:** Implementuj flux-based onset: log compression → adaptive threshold (moving median ~0.2s) → combine z energy-based. Flux jest bardziej robust dla varied dynamics.
**Ref:** RESEARCH §4 "Logarithmic compression of the magnitude spectrum before differencing — applying Γ(X) = log(1 + γ·|X|) — enhances weak transients alongside dominant kicks. An adaptive threshold (moving median over ~0.2 seconds). This method is more robust than pure energy for music with varied dynamics."

### 1.7 Parallelcube threshold calibration
**Problem:** Variance→threshold mapping odbiega od spec: (0→1.7 vs 1.55), (0.02→1.4 vs 1.25). Mniej czuły niż powinien.
**Plik:** `src/hue_visualizer/audio/beat_detector.py:126-133`
**Fix:** Zmień formułę na: `threshold = 1.55 - (variance / 0.02) * 0.30` (clamped [1.25, 1.55]).
**Ref:** RESEARCH §4 "The Parallelcube algorithm uses variance-based adaptive thresholding: map the energy history's variance to a threshold via a linear function between (variance=0, threshold=1.55) and (variance=0.02, threshold=1.25)."

### 1.8 Chase effect
**Problem:** Brak sekwencyjnej aktywacji per-bulb.
**Plik:** nowy mode w `engine.py` lub `spatial.py`
**Fix:** Sequential bulb activation z 50-100ms delay per bulb. Przy 128 BPM i 4 bulbach = ~117ms per bulb na jeden beat.
**Ref:** RESEARCH §7 "Chase: Sequential bulb activation with 50-100ms delay per bulb. At 128 BPM with 4 bulbs, one beat means ~117ms per bulb. Creates directional movement."

### 1.9 Treble sparkle effect
**Problem:** Brak.
**Plik:** nowy w `engine.py`
**Fix:** Map 6-20 kHz energy → brief blue/violet flickers na losowo wybranych bulbach. Symuluje shimmer cymbałów.
**Ref:** RESEARCH §7 "Treble sparkle: Map 6-20 kHz energy to brief blue/violet flickers on randomly selected bulbs. Simulates cymbal shimmer."

### 1.10 Bass pulse effect
**Problem:** Brak explicit bass→red/orange pulse.
**Plik:** `src/hue_visualizer/visualizer/engine.py`
**Fix:** Izoluj 20-250 Hz energy, mapuj na red/orange (hue 0-30°) brightness pulses. Silniejszy bass = jaśniejszy pulse.
**Ref:** RESEARCH §7 "Bass pulse: Isolate 20-250 Hz energy, map to red/orange brightness pulses. Stronger bass = brighter pulse. The quintessential EDM effect."

### 1.11 Palettes powiązane z genre presets
**Problem:** Palettes i genre presets są zdecoupled. Zmiana genre nie zmienia palety.
**Pliki:** `src/hue_visualizer/visualizer/presets.py`, `src/hue_visualizer/server/app.py:373-401`
**Fix:** Dodaj `default_palette` do `GenrePreset`. Aplikuj przy `_apply_genre_preset()`.
**Ref:** RESEARCH §6 "Genre-specific curated palettes: techno (deep purple ~280°, midnight blue ~240°, blood red ~0°), house (warm amber ~35°, coral ~15°, cyan ~180°), trance (ice blue ~200°, white, silver), drum and bass (neon green ~120°, yellow ~60°)."

### 1.12 Intensity selector (Intense / Normal / Chill)
**Problem:** Brak niezależnego od genre kontrolera intensywności.
**Pliki:** frontend + `server/app.py`
**Fix:** 3-level selector wpływający na: transition speed, effect trigger thresholds, flash tau, max brightness, attack/release alphas.
**Ref:** ILS §Effect modes "Intensity selector with three levels: Intense, Normal, and Chill. These affect transition speed, effect trigger thresholds, and how aggressively the algorithm responds to musical events."

### 1.13 Effects size parameter
**Problem:** Wszystkie światła zawsze reagują jednocześnie. Brak kontroli ile świateł zmienia się naraz.
**Plik:** `src/hue_visualizer/visualizer/engine.py`
**Fix:** Parametr effects_size: 1 Light (chase-like), 25%, 50%, 100%. Przy <100% rotuj które światła reagują.
**Ref:** ILS §Spatial effects "Effects Size parameter: '1 Light' (only one light changes at a time), '25%' (a quarter of lights change simultaneously), or '50%' (half). The 100% option drives all lights together."

### 1.14 Light group splitting
**Problem:** Brak podziału świateł na subgrupy z różnymi kolorami/efektami.
**Plik:** `src/hue_visualizer/visualizer/engine.py`
**Fix:** Auto-split na 2-3 subgrupy, każda z innym kolorem z palety i potencjalnie innym efektem.
**Ref:** ILS §Color system "In Standard mode's 'Auto' effects size, the algorithm splits the room's lights into 2-3 subgroups that receive different effects and colors at any given moment."

### 1.15 Entertainment area position data
**Problem:** Light positions hardcoded jako linear 0.0-1.0. Brak integracji z bridge position metadata.
**Pliki:** `src/hue_visualizer/visualizer/spatial.py:31-33`, `src/hue_visualizer/bridge/entertainment_controller.py`
**Fix:** Czytaj x/y/z pozycje z Entertainment API i użyj w spatial mapper.
**Ref:** ILS §Spatial effects "iLightShow does not use the Hue Entertainment API's spatial positioning system — this is the app's most significant unexploited opportunity." RESEARCH §7 "Zone-based mapping: Bass to floor-level or behind-listener bulbs, mids to wall-level, treble to ceiling or front. This leverages the entertainment area position metadata."

---

## P2 — Nice-to-have / Polish

### 2.1 Mel filterbank (32-40 bands)
**Problem:** 7 ręcznie zdefiniowanych rectangular bands. Mel filterbank daje perceptualnie zbalansowaną reprezentację.
**Plik:** `src/hue_visualizer/audio/analyzer.py`
**Ref:** RESEARCH §2 "A Mel filterbank of 32-40 bands produces a perceptually balanced representation ideal for music visualization. LedFx uses 'Melbanks' as its primary frequency feature extraction method."

### 2.2 Confidence scoring z prediction ratio
**Problem:** Confidence oparta na autocorrelation peak value, nie na ratio confirmed/total predictions.
**Plik:** `src/hue_visualizer/audio/beat_detector.py:306`
**Ref:** RESEARCH §4 "Confidence scoring tracks the ratio of confirmed predictions to total predictions over a sliding window. High confidence (>80%) means the system is locked."

### 2.3 Brightness delta limiting
**Problem:** Brak explicit cap na zmianę brightness per klatka. EMA daje implicit limiting ale beat flash bypasuje to.
**Plik:** `src/hue_visualizer/visualizer/engine.py`
**Ref:** RESEARCH §8 "Rate limiting: Clamp maximum brightness change to 30-50% per 100ms frame." §10 "no more than 50% swing in a single 100ms step."

### 2.4 Hysteresis / dead zone
**Problem:** Sub-perceptualne zmiany (<2-3%) wysyłane do bridge powodują unnecessary traffic.
**Plik:** `src/hue_visualizer/visualizer/engine.py`
**Ref:** RESEARCH §8 "Hysteresis: Add a dead zone where changes below 2-3% of full range are suppressed. Prevents subtle fluctuations from causing visible flicker."

### 2.5 User-configurable safe mode (2 Hz)
**Problem:** `max_flash_hz` konfigurowalne z .env ale brak UI toggle. Brak comprehensive safe mode.
**Ref:** RESEARCH §10 "Provide a user-configurable safe mode enforcing these limits with a 2 Hz maximum."

### 2.6 Manual calibration delay
**Problem:** Brak możliwości kompensacji system-specific audio-to-light delay.
**Ref:** ILS §Calibration "Manual calibration lets users set a fixed delay — community consensus suggests +0.3s to +0.6s."

### 2.7 Auto-calibration
**Problem:** Brak automatycznego pomiaru opóźnienia audio→light.
**Ref:** ILS §Effect modes "FX+ Auto-calibration uses the device microphone to measure the actual audio-to-light delay."

### 2.8 Per-light brightness min/max
**Problem:** Wszystkie światła mają ten sam zakres 0.0-1.0.
**Ref:** ILS §User controls "Brightness range: per-light min/max (0-100%), controlling dynamic contrast."

### 2.9 Saturated red → shift hue to orange
**Problem:** Obecna ochrona tylko desaturuje red strobe, nie przesuwa hue na orange.
**Plik:** `src/hue_visualizer/visualizer/engine.py:173-174`
**Ref:** RESEARCH §10 "Never strobe saturated red — substitute orange or pink."

### 2.10 Algorithmic palette generation (complementary, triadic)
**Problem:** Palety ręcznie zdefiniowane. Brak auto-generacji harmonicznych palet.
**Ref:** ILS §Color system "Automatic palettes generate either 2 complementary colors (opposite on color wheel) or 3 triadic colors (evenly spaced at 120° intervals)."

### 2.11 Alternating spatial mode
**Problem:** Brak. Even bulbs = bass/warm, odd = treble/cool.
**Ref:** RESEARCH §7 "Alternating: Even-numbered bulbs respond to one band (bass → warm) while odd bulbs respond to another (treble → cool). Creates contrast."

### 2.12 Wave effect z per-bulb delay
**Problem:** Obecny wave to continuous sine, nie discrete 50-100ms per-bulb delay.
**Plik:** `src/hue_visualizer/visualizer/engine.py:219-225`
**Ref:** RESEARCH §7 "Wave effects: Propagate a brightness or color pulse across bulbs with 50-100ms delay per bulb."

### 2.13 Breathing effect integration
**Problem:** `BreatheEffect` istnieje ale nie jest zintegrowany z engine. Cubic easing zamiast sinusoidal, domyślne parametry odbiegają.
**Pliki:** `src/hue_visualizer/bridge/effects.py:63-95`, `src/hue_visualizer/visualizer/engine.py`
**Ref:** RESEARCH §7 "Breathing: Sinusoidal brightness between ~20% and ~80%. A 4-beat cycle (1.875 seconds at 128 BPM)."

### 2.14 Color cycling z energy modulation
**Problem:** Rotacja hue czysto BPM-driven, brak modulacji prędkości energią.
**Plik:** `src/hue_visualizer/visualizer/engine.py:116-133`
**Ref:** RESEARCH §7 "Color cycling: Speed modulates with energy: halve during breakdowns, double during builds."

### 2.15 Multi-bridge support
**Problem:** Single bridge architecture. Max 10-20 świateł.
**Ref:** ILS §Spatial "users work around this with multiple bridges; one power user runs 85 lights across 5 bridges."

### 2.16 White strobe option
**Problem:** Brak. White maximizes lumen output z Hue bulbs.
**Ref:** ILS §Effect modes "White Strobe replaces colored strobe-at-drops with white-only flashes for higher perceived intensity."

### 2.17 Manual strobe trigger buttons
**Problem:** Brak. Przydatne do budowania energii w konkretnych momentach.
**Ref:** ILS §User controls "Manual strobe triggers: white or color, available any time."

### 2.18 Preset save/load w frontend
**Problem:** Genre buttons apply fixed presets, ale user nie może zapisać custom konfiguracji.
**Ref:** ILS §User controls "Users can save configurations as presets (including music settings, selected lights, and effect options)."

### 2.19 Saturation slider w frontend
**Problem:** Brak kontroli nasycenia kolorów.
**Ref:** ILS §Color system "A saturation slider controls vibrancy — higher values produce intense, saturated colors while lower values yield pastels."

---

## Bugi do naprawienia

### B.1 `_rms_history` list → deque
**Plik:** `src/hue_visualizer/audio/analyzer.py:94,117`
**Fix:** `deque(maxlen=self._rms_window_size)` zamiast `list` z `pop(0)`.

### B.2 Band energy overlap
**Plik:** `src/hue_visualizer/audio/analyzer.py:200-208`
**Fix:** `round()` zamiast `int()` przy obliczaniu bin indices, lub definiuj bands jako contiguous ranges.

### B.3 SpatialMapper dead code
**Plik:** `src/hue_visualizer/visualizer/spatial.py:35-148`
**Fix:** Albo przenieś logikę z engine.py do spatial.py i wywołuj, albo usuń duplicate.

### B.4 `_apply_genre_preset` private attribute access
**Plik:** `src/hue_visualizer/server/app.py:381-400`
**Fix:** Dodaj public settery/metody do `BeatDetector` i `EffectEngine`.

### B.5 Per-light packets zamiast batch
**Plik:** `src/hue_visualizer/server/app.py:196-198`
**Fix:** Batch all light states do jednego `set_input()` call.

### B.6 Bridge HTTP → HTTPS
**Pliki:** `src/hue_visualizer/bridge/entertainment_controller.py:260`, `src/hue_visualizer/bridge/connection.py:27`
**Fix:** `https://` z `verify=False` (self-signed cert na bridge).

### B.7 Frontend: beatStrength never zeroes
**Plik:** `frontend/index.html` (decay `*= 0.92`)
**Fix:** `if (S.beatStrength < 0.001) S.beatStrength = 0;`

### B.8 Frontend: no JSON parse error handling
**Plik:** `frontend/index.html` (`JSON.parse(e.data)`)
**Fix:** Wrap w try-catch.

### B.9 Frontend: no resize debounce
**Plik:** `frontend/index.html`
**Fix:** Debounce `initCanvases()` z 100-200ms timeout.

### B.10 Frontend: server state not synced to UI controls
**Plik:** `frontend/index.html`
**Fix:** Przy połączeniu WS i przy zmianach z serwera, syncuj active states buttonów i slider values.

# Music-to-light synchronization: a complete engineering reference

**Real-time music-reactive lighting is achievable within a web browser controlling Philips Hue bulbs, but the system must be carefully engineered around hard constraints: ~80–120ms end-to-end latency, a practical bulb update ceiling of ~12.5 fps, and photosensitive epilepsy safety limits of 3 Hz maximum flash rate.** The core pipeline — audio capture, FFT analysis, feature extraction, beat detection, effect computation, and Hue Entertainment API streaming — can execute within these bounds when each stage is optimized. This document covers every theoretical and practical element needed to build such a system, from DSP fundamentals through perceptual science to the Hue protocol wire format.

---

## 1. Capturing and processing audio in real time

### Sample rates and buffer mechanics

Digital audio enters the system as a stream of samples at a fixed rate. The two standard rates are **44,100 Hz** (CD-quality, the most common for consumer music) and **48,000 Hz** (professional audio, the default for most browser AudioContexts). Both exceed the Nyquist requirement for human hearing's 20–20,000 Hz range.

Audio arrives in discrete chunks called buffers. Buffer size directly determines minimum latency: **latency = buffer_size / sample_rate**. At 44,100 Hz, a 256-sample buffer fills in ~5.8ms, 512 samples in ~11.6ms, **1,024 samples in ~23.2ms**, and 2,048 samples in ~46.4ms. For music-to-light synchronization, **1,024 samples at 44,100 Hz (~23ms)** is the practical sweet spot — fast enough for perceptible sync, reliable enough to avoid buffer underruns on consumer hardware.

For spectral analysis via the Short-Time Fourier Transform (STFT), frames typically overlap by 50–75%. With a 2,048-sample analysis window and a hop size of 1,024, each frame shares half its data with the adjacent frame. This overlap ensures smooth spectral tracking and prevents transient events from falling between frame boundaries.

### The FFT: from time to frequency

The Fast Fourier Transform converts a time-domain buffer (amplitude over time) into its frequency-domain representation (amplitude at each frequency). It reduces the Discrete Fourier Transform's O(N²) complexity to O(N log N), making real-time operation trivial on modern hardware.

**Frequency resolution** equals sample_rate divided by FFT_size. With 44,100 Hz and an FFT size of 2,048, resolution is **~21.5 Hz per bin** — sufficient to distinguish the fundamental frequencies of most musical elements. Doubling to 4,096 sharpens resolution to ~10.8 Hz but doubles processing time and halves temporal responsiveness. The FFT produces FFT_size/2 useful bins (the upper half mirrors the lower for real-valued signals). With a 2,048-point FFT, that yields **1,024 usable bins** spanning 0 Hz to 22,050 Hz.

Converting a bin index to frequency is straightforward: **frequency = bin_index × (sample_rate / FFT_size)**. At 44,100/2,048, bin 1 ≈ 21.5 Hz, bin 10 ≈ 215 Hz, bin 100 ≈ 2,153 Hz, and bin 1,024 = 22,050 Hz (Nyquist).

### Windowing: taming spectral leakage

Extracting a finite audio segment for FFT creates discontinuities at the boundaries, causing spectral leakage — energy from a single frequency smearing across neighboring bins. Window functions taper the signal smoothly to zero at the edges before the FFT is applied.

The **Hann window** (raised cosine) is the recommended default for music analysis. National Instruments notes it is "satisfactory in 95 percent of cases." It provides a good balance between frequency resolution and leakage suppression. The **Hamming window** is similar but retains a small discontinuity at the edges, offering slightly better nearest-sidelobe attenuation at the cost of distant-sidelobe roll-off — more common in speech processing. The **Blackman window** achieves excellent sidelobe suppression (~−58 dB) with the widest main lobe, useful when dynamic range is critical but frequency resolution can be sacrificed.

For general music visualization, **always use Hann**.

### Power spectrum vs magnitude spectrum

The magnitude spectrum is the absolute value of the complex FFT output: |X(k)|. The power spectrum is its square: |X(k)|². The power spectrum emphasizes strong-vs-weak component differences and is standard for energy-per-band calculations. The magnitude spectrum is more perceptually intuitive and preferred for direct visualization. For energy calculations throughout this document, use the power spectrum (sum of squared magnitudes).

**Zero-padding** — appending zeros to the frame before FFT — increases the number of output bins and smooths the spectral envelope by interpolation. It does not improve true frequency resolution (still determined by actual window length) but helps with precise peak location.

---

## 2. Decomposing the spectrum into musically meaningful bands

### Seven standard frequency bands

Different frequency regions correspond to distinct musical elements and perceptual qualities. The following division is standard in audio engineering:

- **Sub-bass (20–60 Hz):** More felt than heard. Kick drum fundamentals (~40–60 Hz), sub-bass synthesizer tones. In a 44,100/2,048 FFT, this spans approximately bins 1–3.
- **Bass (60–250 Hz):** Core of the rhythm section. Bass guitar, bass synth body, kick drum tonal weight. Bins 3–12.
- **Low-mid (250–500 Hz):** Warmth and body. Lower harmonics of guitars, piano. Excess energy causes "muddiness." Bins 12–23.
- **Mid (500–2,000 Hz):** Core melodic range. Vocals, lead instruments, most melodic content. Bins 23–93.
- **Upper-mid (2,000–4,000 Hz):** The presence range. Vocal consonants, percussive attack, guitar "bite." Human hearing peaks here. Bins 93–186.
- **Presence (4,000–6,000 Hz):** Definition, clarity, sibilance, cymbal body. Bins 186–279.
- **Brilliance (6,000–20,000 Hz):** Air, shimmer, sparkle. Cymbal sizzle, breath, string sheen. Bins 279–930.

Notice the dramatic imbalance: sub-bass has ~2 bins while brilliance has over 650. This is because FFT bins are linearly spaced in frequency, but **human pitch perception is logarithmic** — each octave doubles in frequency. The interval from 20 to 40 Hz (one octave) is musically as significant as 5,000 to 10,000 Hz, yet the latter spans far more bins. Logarithmic frequency grouping corrects this by assigning perceptually equal weight to each octave.

To calculate band energy: **sum the squared magnitudes of all FFT bins within the band's frequency range**. For display, convert to decibels (10 × log₁₀ of the energy) or take the square root for an RMS-like value.

### The Mel scale: perceptual frequency warping

The Mel scale warps frequency to match human pitch perception. The conversion formula is: **mel = 2595 × log₁₀(1 + f/700)**. The scale is approximately linear below 1,000 Hz and logarithmic above it. At 100 Hz, the Mel value is ~150; at 1,000 Hz, ~1,000 (the reference point); at 10,000 Hz, ~3,057.

A Mel filterbank consists of triangular bandpass filters spaced uniformly on the Mel scale — narrow and densely packed at low frequencies, progressively wider at high frequencies. Mapping 1,024 linear FFT bins through a Mel filterbank of **32–40 bands** produces a perceptually balanced representation ideal for music visualization. LedFx, the leading open-source music-to-LED system, uses "Melbanks" — Mel-frequency-scaled filterbanks at three resolutions — as its primary frequency feature extraction method.

---

## 3. Amplitude, spectral features, and onset detection

### RMS energy and peak detection

**RMS (Root Mean Square)** energy is the standard measure of signal amplitude: RMS = √(1/N × Σ xᵢ²). It correlates well with perceived loudness and provides a stable "how loud right now" signal. **Peak amplitude** — the maximum absolute sample value in a frame — responds instantly to transients (drum hits) but is noisy and overreactive. The best approach combines both: **peak for fast triggers, RMS for sustained level**.

### Envelope following

An envelope follower tracks the amplitude contour with separate attack and release rates. When the signal rises above the current envelope, the envelope climbs quickly (attack); when the signal drops, the envelope falls slowly (decay). This is implemented via **exponential moving average (EMA)**: `smoothed = α × new_value + (1 − α) × previous_smoothed`.

For music-reactive lighting, use asymmetric alpha values: **attack α ≈ 0.5–0.8** (lights snap on with the beat) and **release α ≈ 0.05–0.15** (lights fade gracefully). At 60 fps with α_attack = 0.7 and α_release = 0.07, the system snaps to a beat in 2–3 frames (~33–50ms) but takes 15–20 frames (~250–330ms) to fade — producing the characteristic "punch and glow" that feels musical.

### Spectral centroid, flux, and rolloff

**Spectral centroid** is the "center of mass" of the spectrum: **Centroid = Σ(frequency_k × |X(k)|) / Σ(|X(k)|)**. It indicates perceived brightness — a high centroid (~4,000–8,000 Hz) means bright, treble-heavy sound (cymbals), while a low centroid (~200–500 Hz) means dark, bass-heavy sound. For lighting, centroid **directly drives hue position**: low centroid → warm colors, high centroid → cool colors. This provides a single, continuous value that naturally tracks timbral character.

**Spectral flux** measures spectral change between frames: **Flux = Σ max(0, |X_current(k)| − |X_previous(k)|)**. By considering only positive differences (half-wave rectification), it detects energy increases — exactly what characterizes onsets like drum hits and note attacks. High flux = sudden spectral change; near-zero = steady state. Research by Dixon (2006) and Bello et al. (2005) confirms spectral flux achieves onset detection accuracy comparable to more complex methods across diverse musical styles.

**Spectral rolloff** is the frequency below which a specified percentage (typically 85%) of total spectral energy is concentrated. A low rolloff (~1,000 Hz at 85%) indicates bass-heavy passages; a high rolloff (~8,000 Hz) indicates broadband or treble-rich content. Useful for distinguishing atmospheric breakdowns from energetic drops.

### Onset detection

Onsets — the beginnings of new musical events — trigger the most impactful light effects. The **spectral flux method** is the recommended approach: compute the onset detection function (sum of positive spectral differences) per frame, then apply **adaptive thresholding**. The threshold is set as the local moving median (over 5–10 frames) multiplied by a constant (1.3–1.5×). Moving median is preferred over moving average because it resists inflation from outlier peaks. A **minimum inter-onset interval of 50–100ms** prevents double-triggering.

For richer effects, perform onset detection **separately per frequency band**: low-band onsets (kicks) trigger bass pulses, mid-band onsets (snares) trigger white flashes, high-band onsets (hi-hats) trigger sparkle effects on random bulbs. This frequency-aware triggering creates deep musical-visual correspondence.

---

## 4. Beat detection and tempo tracking

### Energy-based beat detection step by step

The simplest real-time approach monitors bass energy frame by frame. For each frame (~23ms at 1024 samples/44,100 Hz), compute the FFT and isolate the **60–200 Hz band** where EDM kick drums concentrate their energy. Sum the squared magnitudes of the relevant bins (approximately bins 1–9 at 2,048 FFT/44,100 Hz). Store this value in a history buffer of ~43 frames (~1 second). Compute the running average. A beat is declared when instantaneous bass energy exceeds the average by a **multiplicative factor of 1.3× to 1.5×**.

The optimal threshold is not static. The Parallelcube algorithm uses **variance-based adaptive thresholding**: map the energy history's variance to a threshold via a linear function between (variance=0, threshold=1.55) and (variance=0.02, threshold=1.25). Higher variance (noisy mix) demands a lower threshold; lower variance (clean EDM kick) permits a higher threshold.

Enforce a **minimum inter-beat interval of ~300ms** as a refractory period. At 120 BPM, beats are 500ms apart; at 150 BPM, 400ms; at 200 BPM (practical upper limit), 300ms. This prevents double-triggering from kick drum tails.

### Spectral flux beat detection

Rather than absolute energy, spectral flux tracks **rate of spectral change**. Logarithmic compression of the magnitude spectrum before differencing — applying Γ(X) = log(1 + γ·|X|) with γ ≥ 1 — enhances weak transients (hi-hats, snares) alongside dominant kicks. An adaptive threshold (moving median over ~0.2 seconds) is applied to the flux signal, with peaks selected at minimum 40ms distance. This method is **more robust than pure energy for music with varied dynamics** because it responds to change rather than absolute level.

### BPM estimation via autocorrelation

After computing an onset detection function, its autocorrelation reveals the dominant repeating period. Compute the autocorrelation r(τ) = Σ p(k+τ)·p(k) over lags corresponding to **60–200 BPM** (periods of 300ms to 1 second). The lag with the highest peak gives the beat period; **BPM = 60 / period_in_seconds**.

A critical challenge is **octave error**: a 128 BPM track may register as 64 or 256 BPM. Constraining the search range to expected genre BPM ranges mitigates this. The Percival-Tzanetakis algorithm (2014) addresses this through "generalized autocorrelation" with magnitude compression and cross-correlation scoring against ideal pulse trains.

### The comb filter approach

Eric Scheirer's method (1998) uses parallel comb filters tuned to candidate tempos. Audio is split into six octave bands, envelopes are extracted, onsets are derived via half-wave-rectified differentiation, and a bank of comb filters at each candidate BPM is applied. The resonator with maximum output energy indicates the dominant tempo. Comb filters are inherently causal, accumulate evidence naturally, and handle syncopation well. However, they offer limited resolution and can lock onto sub/harmonics of the true tempo.

### Beat tracking vs beat detection

**Beat detection** identifies individual beat events reactively — each energy spike triggers a flag. **Beat tracking** maintains a continuous model of tempo and phase, predicting where future beats will fall. For light synchronization, **tracking is far superior** because it enables anticipation: lights can be triggered slightly before the beat arrives, compensating for system latency.

The **phase-locked loop (PLL) concept** treats music as a quasi-periodic signal. Once an initial BPM estimate is established, an internal oscillator predicts the next beat as: **next_beat = last_beat_time + beat_period**. When a detected onset arrives earlier than predicted, the tempo estimate increases; when later, it decreases. The correction is proportional: **phase_correction = α × timing_error** (α between 0.1 and 0.5). This is inherently self-correcting — small timing errors are gradually absorbed.

**Confidence scoring** tracks the ratio of confirmed predictions to total predictions over a sliding window. High confidence (>80%) means the system is locked and can rely on prediction for latency compensation. Low confidence triggers fallback to reactive-only mode.

### Handling EDM structure

EDM follows a highly structured macro-form: Intro → Buildup → Drop → Breakdown → Buildup → Drop → Outro. The EDMFormer research (2025) achieved 88.3% per-frame section classification accuracy, confirming these transitions are defined by energy, rhythm, and timbre — all detectable in real time.

**Drops** produce a sudden, massive bass energy spike (>3× running average) following a period of reduced bass. **Buildups** show rising overall energy, increasing hi-hat density, snare rolls, and pitch-ascending riser effects (detectable as sustained increase in spectral centroid). **Breakdowns** exhibit near-zero energy below 200 Hz with sustained mid-to-high content — the four-on-the-floor kick disappears.

**Half-time vs double-time** is critical for dubstep: at 140 BPM, the drop feels at 70 BPM due to half-time drum patterns. Maintain awareness that harmonic tempo remains 140 BPM; don't reset the BPM estimate when the rhythmic accent shifts.

### Downbeat detection

Identifying beat "1" of a bar enables 4-bar or 8-bar lighting patterns. In EDM's 4/4 time, the kick typically accents beats 1 and 3 while the snare hits beats 2 and 4. By detecting the spectral oscillation between low-frequency energy (kicks) and mid-frequency energy (snares), bar position can be inferred. Perfect real-time downbeat detection remains an open problem. A practical heuristic: once beat tracking stabilizes, group beats into fours and look for the strongest bass accent to mark beat 1. Being off by one beat is acceptable for most lighting applications since visual patterns repeat each bar.

### Key timing reference

At **128 BPM** (the most common EDM tempo), beats fall every **468.75ms**. A bar of 4 beats = **1.875 seconds**. An 8-bar phrase = **15 seconds**. A 16-bar phrase = **30 seconds**. At 140 BPM: **~428.6ms** per beat. At 174 BPM (drum and bass): **~344.8ms** per beat. These phrase-level timings map to macro lighting dynamics — palette changes, effect mode switches, and intensity shifts.

---

## 5. How existing tools and libraries approach beat detection

**Aubio** (C library with Python bindings) is designed for real-time causal processing. It provides onset detection via energy, HFC, complex domain, phase, and spectral flux methods, plus beat tracking via Matthew Davies' adaptively weighted comb filterbank algorithm. A hop size of 512 at 44,100 Hz yields ~86 detection cycles per second. Aubio's beat tracker "prefers measurements around 107 BPM" (minimum 40, maximum 250), which can cause octave errors with drum and bass.

**Essentia** (Music Technology Group, Barcelona) offers RhythmExtractor2013 with "multifeature" (accurate, slow) and "degara" (fast) modes, plus PercivalBpmEstimator implementing the autocorrelation algorithm. TempoCNN provides deep-learning-based tempo estimation. Not designed for real-time streaming but useful for reference.

**Madmom** (Python) represents the research frontier with RNNBeatProcessor (recurrent neural network beat activation), DBNBeatTrackingProcessor (dynamic Bayesian network decoding), and CombFilterTempoHistogramProcessor. Computationally heavier than aubio.

**DJ software** (Traktor, Rekordbox) performs offline beat analysis when tracks are imported, using waveform transient analysis and assuming 4/4 time. For EDM with a strong four-on-the-floor kick, automatic analysis is highly reliable (correct ~95% of the time). These systems represent the gold standard.

**Spotify's Audio Analysis API** (now partially deprecated as of late 2024) provided pre-computed beat, bar, section, tatum, and segment data with precise timestamps. iLightShow and LedFx v3 leverage this data for "zero-latency synchronization" — they know beat timestamps and drop locations in advance, enabling anticipatory effects.

---

## 6. Color theory for music-reactive lighting

### Color models and Hue-specific constraints

**HSB/HSV** is the most intuitive model for music mapping. Hue (0–360°) maps cleanly to frequency content, Saturation (0–100%) to spectral purity, and Brightness/Value (0–100%) to energy/amplitude. Incrementing hue alone produces smooth rainbow sweeps impossible to achieve intuitively with interdependent RGB channels.

**CIE 1931 xy chromaticity** is the native color space of the Hue API. Colors are specified as two chromaticity coordinates (x, y) plus brightness. The Hue bridge accepts colors via `xy` (CIE coordinates), `ct` (color temperature in mireds), or `hue`/`sat` (16-bit hue 0–65535, 8-bit saturation 0–254). All modern Hue color products use **Gamut C** — Red (0.692, 0.308), Green (0.17, 0.70), Blue (0.153, 0.048). Colors outside this triangle are clamped to the nearest boundary point.

The recommended workflow: **compute colors in HSV** (intuitive for mapping), convert HSV → RGB, apply gamma correction (sRGB, gamma ~2.2), transform RGB → CIE XYZ via a wide-gamut matrix, then project to xy coordinates. Clamp to the target bulb's gamut before transmission.

### Mapping frequency to color

The natural mapping treats the audio spectrum like the light spectrum: **bass → red/warm, treble → blue/cool**. This aligns with cross-modal research showing non-synesthetes consistently associate lower pitches with darker, warmer colors and higher pitches with lighter, cooler ones.

A practical four-band frequency-to-hue mapping:

- **Bass (20–250 Hz)** → red/orange, hue 0–30°: kick drums, bass lines, sub-bass
- **Low-mids/mids (250–2,000 Hz)** → yellow/green, hue 30–150°: vocals, melodies, snares
- **Upper-mids (2–6 kHz)** → blue/cyan, hue 150–240°: vocal presence, hi-hat body
- **Highs (6–20 kHz)** → purple/violet, hue 240–300°: cymbals, air, overtones

Alternatively, the **spectral centroid can directly drive hue** as a single continuous value, naturally shifting warmer as bass dominates and cooler as treble dominates. This is more fluid than discrete band assignment. Map the log-scaled centroid to hue 0–300° for smooth, moment-to-moment timbral color tracking.

**Genre-specific curated palettes** offer another approach: techno (deep purple ~280°, midnight blue ~240°, blood red ~0°), house (warm amber ~35°, coral ~15°, cyan ~180°), trance (ice blue ~200°, white, silver), drum and bass (neon green ~120°, yellow ~60°). These can override or modulate the frequency-based mapping.

### Energy to brightness and saturation

Map **RMS energy to brightness** using a logarithmic or power curve (gamma ~2.0–2.5) to match human visual perception (Weber-Fechner law). Without this correction, low-energy passages appear too dim and high-energy passages show no variation. The formula: `perceived_brightness = (RMS / max_RMS)^(1/gamma) × max_brightness`.

**Spectral flatness** (ratio of geometric mean to arithmetic mean of the spectrum) can inversely drive saturation: a flat, noise-like spectrum → low saturation (pastel/white); a peaked, tonal spectrum → high saturation (vivid color). A strong, pure bass drop produces maximum saturation; a dense, layered breakdown produces softer pastels.

### Color transitions and interpolation

**HSV interpolation** is superior to RGB interpolation. Linear RGB between complementary colors passes through desaturated gray at the midpoint — cyan (0, 255, 255) to red (255, 0, 0) in RGB goes through gray (127, 127, 127). In HSV, interpolating hue, saturation, and brightness independently maintains saturation throughout.

However, hue is circular (0° = 360°), so the system must choose shortest vs longest path. Interpolating from blue (240°) to red (0°): the shortest path goes through magenta (240° → 300° → 360°, a 120° arc); the longest goes through cyan, green, and yellow (240° arc). **Almost always use the shortest path** unless deliberately sweeping the rainbow. Implementation: if absolute hue difference exceeds 180°, add or subtract 360° before interpolating.

**Transition timing** should sync to musical divisions. At 120 BPM (500ms per beat): full-beat transitions (500ms) for mood shifts, half-beat (250ms) for rhythmic color changes, quarter-beat (125ms) for rapid effects. **Exponential decay** (`brightness × e^(-t/τ)` with τ = 200–500ms) is ideal for beat-triggered flash-and-fade.

---

## 7. Light effect design patterns

### The core effect vocabulary

**Pulse on beat**: The foundational effect. On beat detection, spike brightness to 100%, decay via exponential function over 200–300ms. Attack time should be 0–50ms. The Hue API's default 400ms transition is too slow — set `transitiontime` to 0 for attack, then issue a follow-up with longer transition for decay.

**Color cycling**: Continuously increment hue tied to BPM. At 128 BPM, a full 360° rotation over 16 beats (~7.5 seconds) produces a gentle musical rainbow. Speed modulates with energy: halve during breakdowns, double during builds.

**Strobe**: Rapid on-off flashing, constrained by both Hue's rate limit (~10 REST commands/second, ~25 Entertainment API updates/second) and **epilepsy safety (maximum 3 Hz at full brightness delta)**. Practical Hue maximum: 2–3 flashes per second. Never strobe with saturated red.

**Breathing**: Sinusoidal brightness between ~20% and ~80%. A 4-beat cycle (1.875 seconds at 128 BPM) is natural — inhale (brighten) on beats 1–2, exhale (dim) on beats 3–4. Ideal for breakdowns and ambient passages.

**Chase**: Sequential bulb activation with 50–100ms delay per bulb. At 128 BPM with 4 bulbs, one beat means ~117ms per bulb. Creates directional movement.

**Bass pulse**: Isolate 20–250 Hz energy, map to red/orange brightness pulses. Stronger bass = brighter pulse. The quintessential EDM effect.

**Treble sparkle**: Map 6–20 kHz energy to brief blue/violet flickers on randomly selected bulbs. Simulates cymbal shimmer.

### The comprehensive audio-to-light mapping

**Beat detection** drives brightness pulses (spike to 100%, exponential decay 200–300ms) and strobe triggers. **Bass energy (20–250 Hz)** drives warm hues (0–30°) and brightness boost. **Mid energy (250–2,000 Hz)** maps to green/yellow (60–120°) with saturation proportional to band dominance. **Treble energy (6–20 kHz)** maps to cool hues (200–280°) with higher saturation for stronger signals. **Spectral centroid** continuously positions the overall hue (low centroid → warm, high centroid → cool). **Onset/transient detection** triggers instantaneous brightness spikes. **RMS energy envelope** controls overall brightness via logarithmic mapping. **BPM** sets transition speeds and cycling periods. **Spectral flux** drives color change rate — high flux = rapid hue shifts; low flux = stable color.

### Multi-bulb spatial choreography

**Wave effects**: Propagate a brightness or color pulse across bulbs with 50–100ms delay per bulb. A 5-bulb array completes a wave in 200–500ms. Both brightness waves and hue waves work well.

**Alternating**: Even-numbered bulbs respond to one band (bass → warm) while odd bulbs respond to another (treble → cool). Creates contrast without requiring spatial position data.

**Zone-based mapping**: Bass to floor-level or behind-listener bulbs, mids to wall-level, treble to ceiling or front. Spatializes the frequency content for an immersive experience. This leverages the entertainment area position metadata in the Hue API.

**Mirror effects**: For symmetric arrangements, mirror patterns around the center — bulbs 1 and 6 are identical, 2 and 5 identical. Produces balanced, professional results.

### Adapting to song sections

By tracking energy, centroid, onset rate, and spectral flux over 8–32 beat windows, the system distinguishes sections automatically:

**Buildup**: Rising RMS energy, rising spectral centroid, increasing onset density. Response — gradually increase brightness, shift toward warmer/saturated hues, ramp pulse intensity.

**Drop**: Sudden bass spike (>3× running average) after energy dip. Response — flash all bulbs to maximum brightness with saturated warm color, settle into intense bass-reactive pattern.

**Breakdown**: Reduced RMS, absent kick, low spectral centroid. Response — dim to 20–40%, shift to cool pastels, slow transitions, breathing effects, minimal beat pulses.

### The hybrid reactive-generative approach

Pure **reactive** lighting (direct audio-to-light mapping) produces tight sync but boring lights during quiet passages and chaos during dense ones. Pure **generative** lighting (autonomous patterns) looks good but disconnects from music. The optimal strategy is **hybrid**: a generative base layer (slow hue rotation completing one cycle every 30–60 seconds, gentle breathing at 0.25 Hz, gradual spatial waves) modulated by reactive triggers (beat-triggered brightness pulses, bass-driven hue shifts, drop overrides). During quiet passages, the generative base dominates (~80%); during energetic sections, reactive effects take over (~80%). This produces lighting that is always visually appealing and always musically connected.

---

## 8. Smoothing, normalization, and anti-flicker techniques

### Parameter smoothing

Raw audio features jitter rapidly. Direct mapping to light output produces unpleasant flickering. **Exponential moving average (EMA)** is the fundamental smoothing operation: `smoothed = α × raw + (1 − α) × previous_smoothed`.

Recommended alpha values by parameter:
- **Brightness**: α ≈ 0.3–0.5. Must be responsive to beats but stable enough to avoid jitter.
- **Color (hue)**: α ≈ 0.05–0.15. Rapid hue jumps look chaotic; color should evolve gradually. At α = 0.1 and 30 fps, convergence takes roughly 1 second.
- **Saturation**: α ≈ 0.1–0.2. Similar to color — moderate smoothing.

**Asymmetric attack/release** is crucial: use α_attack ≈ 0.5–0.8 when the value is rising (lights snap on) and α_release ≈ 0.1–0.2 when falling (lights fade gracefully). This mimics audio compressor behavior and produces the most musical-feeling light response.

### Dynamic normalization and auto-gain

Without normalization, turning the volume knob changes the visual experience. **Sliding window normalization** maintains a 5–10 second rolling history of RMS. The current value is normalized against the window's mean and range: `normalized = (current − window_min) / (window_max − window_min)`. A 5-second window (~10 beats at 120 BPM) adapts to different tracks while remaining stable over individual beats. WLED Sound Reactive implements four AGC presets: Off, Normal (smooth following), Vivid (quick adjustment), and Lazy (slower, better for equalized displays).

All beat detection should use **relative thresholds** — current energy vs recent average ratio — not absolute dB levels. This makes the system volume-independent.

### Anti-flicker techniques

**Rate limiting**: Clamp maximum brightness change to 30–50% per 100ms frame. For hue: maximum 30–60° per frame unless a deliberate flash is triggered.

**Hysteresis**: Add a dead zone where changes below 2–3% of full range are suppressed. Prevents subtle fluctuations from causing visible flicker, especially important given Hue bulbs' own response characteristics.

**Minimum transition times**: Never send successive brightness commands closer than the bulb's effective response time (~80ms for Hue Entertainment API). Let the bridge's built-in interpolation do the smoothing work.

---

## 9. Philips Hue platform: capabilities, constraints, and the Entertainment API

### The standard REST API is inadequate for music sync

The Hue REST (CLIP) API limits individual light commands to **~10 per second** to the bridge total, and group commands to **~1 per second**. Controlling 5 lights means each updates only twice per second — far too slow. Each command involves HTTP/HTTPS overhead with **30–100ms round-trip latency**. Signify's official developer documentation explicitly states: "the REST API must not be used for use cases with continuous fast light control."

### The Entertainment API: real-time streaming via DTLS

The Entertainment API is a **UDP-based streaming protocol** designed for exactly this use case. It uses **DTLS 1.2** (Datagram Transport Layer Security, the UDP equivalent of TLS) on port **2100**, with Pre-Shared Key authentication using cipher suite `TLS_PSK_WITH_AES_128_GCM_SHA256`. The PSK identity is the application username; the PSK value is the `clientkey` generated during initial bridge pairing.

The bridge converts incoming UDP packets to Zigbee messages at a maximum rate of **25 Hz** (one every 40ms). Philips recommends sending at **50–60 Hz** to compensate for UDP packet loss — the bridge decimates to 25 Hz internally. Developer reports confirm the **effective visible update rate at bulbs is approximately 12.5 fps** due to Zigbee timing and bulb processing.

The streaming packet is binary, big-endian: a 16-byte header ("HueStream", API version, color mode byte — 0x00 for RGB, 0x01 for XY+brightness), the 36-byte entertainment configuration UUID, then **7 bytes per channel** (1-byte channel ID plus three 16-bit color values, 0–65535). All lights in the entertainment group receive simultaneous updates in a single packet. A 10-light packet is only 122 bytes — easily within UDP limits.

The V2 API supports up to **20 light channels per entertainment configuration** (up from 10 in V1). Only **one entertainment stream can be active per bridge** at any time. For larger installations, multiple bridges are required (iLightShow has been tested with up to 12 bridges and 120+ lights).

### Latency budget: audio to photon

| Stage | Typical Latency |
|---|---|
| Audio capture (1024-sample buffer at 44.1 kHz) | ~23ms |
| FFT + feature extraction + beat detection | 1–5ms |
| Application logic (effect computation) | <1ms |
| Network to Hue bridge (local WiFi UDP) | 2–10ms |
| Bridge processing + Zigbee queue | 2–40ms |
| Zigbee transmission to proxy bulb + broadcast | 6–20ms |
| Bulb internal processing + LED response | 5–25ms |
| **Optimistic total** | **~40–80ms** |
| **Typical total** | **~80–120ms** |
| **Pessimistic (WiFi congestion, large buffers)** | **~150–250ms** |

Human audio-visual sync tolerance is **~80–100ms** — at the typical 80–120ms pipeline, the system sits at the edge of perceptibility. Beat effects will feel "close but slightly late." Mitigation strategies: **predictive triggering** (send commands ~80–100ms before predicted beat), **beat prediction via PLL** (once BPM is locked), and prioritizing **brightness changes** (most time-critical) over color changes (where latency matters less).

### Working within Hue's constraints

The bridge smoothly interpolates between consecutive commanded states. If color A is sent at time 0 and color B at time 40ms, the bulb ramps smoothly between them. This means **visually smooth gradients are achievable even at 12.5 fps** — the bulb's firmware handles interpolation. Design effects that look good at this frame rate; avoid anything requiring >25 distinct states per second.

**Prioritize update types**: brightness modulation for rhythmic elements (kick, snare) — most perceptibly time-critical. Color changes for harmonic elements (chord changes, mood shifts) — tolerate more latency. If the audio analysis runs faster than 25 Hz (e.g., ~43 Hz at 1024-sample hops), maintain a "current target state" buffer updated by the analysis thread, read by a separate sender thread at the bridge's rate. Use **peak-hold logic** to avoid losing transient peaks between sends.

### Web application architecture constraint

Browsers **cannot send raw UDP or DTLS packets** — there is no browser API for this. The Entertainment API therefore requires a **local proxy service**: a small server-side application (Node.js, Python, etc.) that receives WebSocket messages from the browser and forwards them as DTLS packets to the Hue bridge. The architecture becomes: **Browser (Web Audio API → feature extraction → WebSocket) → Local proxy (WebSocket → DTLS) → Hue Bridge → Bulbs**. The proxy can run on the same machine. For the standard REST API (lower performance but no proxy needed), `fetch()` works directly from the browser.

---

## 10. Perceptual science: why synchronization matters and how humans experience it

### Audio-visual synchrony perception

Humans detect audio-visual delays as short as **20ms** in controlled experiments, but the "synchrony window" — the Temporal Binding Window within which events are perceived as simultaneous — extends to approximately **200ms**. For rhythmic music, the practical threshold where misalignment becomes noticeable is **~80–100ms**; above **~150ms**, the illusion of synchronization breaks entirely.

The window is **asymmetric**: visual-leading asynchronies (light before sound) are harder to detect than audio-leading ones. This means if residual lag exists, it is better for light to arrive **slightly early** rather than late — a useful property for beat-prediction systems that trigger lights ahead of time.

Cross-modal research demonstrates that synchronized lights genuinely enhance the perceived musical experience. The brain integrates simultaneous audio-visual events through Bayesian integration: a light pulse coinciding with a beat makes both the beat feel more powerful and the flash more vivid. This "temporal binding" engages deep multisensory mechanisms — the effect is neurological, not merely aesthetic. Poorly synchronized lights (even by 100–200ms) actively diminish the experience.

### Brightness perception follows logarithmic laws

The **Weber-Fechner law** states that perceived sensation is proportional to the logarithm of physical stimulus intensity. For brightness: doubling perceived brightness requires approximately quadrupling physical light output. Stevens' Power Law refines this with an exponent of ~0.33–0.5 for brightness perception.

The practical consequence is critical: **a linear 0–255 brightness value does NOT produce linearly perceived brightness**. Apply **gamma correction** (power curve, typically gamma 2.2–2.5) when mapping audio energy to brightness. Without correction, low-energy passages look too dim, transitions feel abrupt, and high-energy passages show no variation. The formula: `output = (input / max)^gamma × max`.

### Fletcher-Munson curves and frequency weighting

Human hearing sensitivity varies dramatically with frequency. The ear is most sensitive at **2–5 kHz** and least sensitive at very low and very high frequencies. A 50 Hz bass tone must be approximately **40 dB louder** than a 2,000 Hz tone to sound equally loud (at the 10-phon level). This means raw FFT magnitudes **significantly underrepresent the perceived energy of bass frequencies** relative to midrange.

For music-light systems, apply frequency-dependent boost to bass band energies — empirically, multiplying bass energy by **1.5–3×** relative to mids matches perceptual salience. Without this, a thunderous bass drop that shakes the room will produce a weaker light response than a much quieter mid-frequency element.

### EDM subgenre BPM and energy profiles

| Subgenre | BPM Range | Character |
|---|---|---|
| Ambient / Downtempo | 60–100 | Slow, atmospheric |
| Deep House | 110–125 | Soulful, laid-back |
| House | 118–130 | Four-on-the-floor groove |
| Melodic Techno | 120–126 | Atmospheric, arpeggiated |
| Progressive House | 125–135 | Building, evolving |
| Techno | 125–145 | Mechanical, loop-based |
| UK Garage / 2-Step | 130–140 | Syncopated, shuffled |
| Trance | 128–145 | Euphoric build-and-release |
| Dubstep | 138–142 (half-time ~70) | Heavy sub-bass, half-time drops |
| Hardstyle | 145–155 | Pitched/reverse kicks |
| Drum and Bass | 160–180 | Fast breakbeats, deep sub-bass |
| Hardcore / Gabber | 160–200 | Very fast, saturated kicks |

Lower BPM genres call for slow, smooth transitions and gentle brightness modulation. Mid-range (house, techno) benefits from beat-synced pulses at ~2.1 Hz — energetic but not frantic. High-BPM genres (DnB at 170+ BPM ≈ 2.8 beats/sec) risk epilepsy concerns if pulsing on every beat; trigger on every other beat or on downbeats only.

### Photosensitive epilepsy safety limits

Photosensitive epilepsy affects approximately **1 in 4,000 people**. Seizures are triggered most commonly at **3–30 Hz flash rates**, with the peak danger zone at **15–25 Hz**. Red flashes are specifically identified as highest risk — the 1997 Pokémon incident (red-blue alternation at 12.5 Hz) caused seizures in 685 children. Closing eyes does not help because red light passes through eyelids.

**Mandatory safety constraints for the system:**
- Never flash faster than **3 Hz** at full brightness range (minimum 333ms between full on-off cycles)
- Limit maximum brightness delta per update (no more than 50% swing in a single 100ms step)
- **Never strobe saturated red** — substitute orange or pink
- Implement a debounce cooldown of at least 120ms between high-intensity events
- Provide a user-configurable **safe mode** enforcing these limits with a 2 Hz maximum
- Keep transition times at 200ms or longer for brightness pulses
- Hue's ~12.5 fps effective rate naturally limits the fastest possible flash rate, but patterns at 3–12 Hz are achievable and still dangerous

---

## 11. System architecture: the complete pipeline

### The seven-stage processing chain

The standard architecture, validated across LedFx, WLED Sound Reactive, and multiple community projects, follows this pipeline:

**Stage 1 — Audio Input**: Microphone capture into a ring buffer. In a web app, use the Web Audio API's `getUserMedia` for mic access and `AnalyserNode` for built-in FFT. Configure with `fftSize: 2048` and `smoothingTimeConstant: 0` (handle smoothing manually for more control).

**Stage 2 — Windowing**: Apply Hann window to the current frame. The AnalyserNode handles this internally; for AudioWorklet custom processing, apply explicitly.

**Stage 3 — FFT**: Transform to frequency domain. AnalyserNode provides `getFloatFrequencyData()` (in dB) or `getByteFrequencyData()` (unsigned 8-bit). For custom processing, use an FFT library or WebAssembly.

**Stage 4 — Feature Extraction**: Compute band energies (seven bands), spectral centroid, spectral flux, spectral rolloff, and RMS. These are the raw audio features that drive all downstream effects.

**Stage 5 — Beat Detection**: Analyze bass energy against adaptive threshold, maintain BPM estimate via autocorrelation, run PLL for beat prediction. Output: beat events, predicted next beat time, current BPM, confidence score.

**Stage 6 — Effect Engine**: Map audio features to light parameters through configurable effects. Multiple effect layers blend via additive, multiplicative, or maximum operations. Apply smoothing (asymmetric EMA), gamma correction, and safety limiting.

**Stage 7 — Output**: Format light states and transmit. For Hue Entertainment API: serialize as binary streaming packets and send via WebSocket to local proxy, which forwards as DTLS/UDP to the bridge. For REST API fallback: send HTTP PUT commands via `fetch()` directly.

This pipeline runs **once per audio frame (~23ms at 1024-sample hops)**, producing ~43 updates per second internally even though the output stage transmits at the bridge's 25 Hz ceiling.

### Effect engine layering

The effect engine should support **multiple simultaneous layers** with blending. A base layer provides continuous generative patterns (slow hue rotation, breathing). A reactive layer overlays beat-triggered brightness pulses. A section-detection layer modulates overall intensity and palette. Blending modes: **maximum** (for overlaying beat flashes on a base), **multiplicative** (for applying brightness envelopes), **additive** (for combining independent color contributions). Each layer outputs per-bulb HSV values; the blender combines them into final RGB values for transmission.

### Web-specific implementation notes

Use **`requestAnimationFrame`** for the main render/update loop — it syncs with display refresh (~60 Hz), auto-pauses when the tab is hidden, and provides accurate timestamps. For audio analysis polling, rAF at 60 fps is sufficient. For heavy feature extraction or beat detection beyond AnalyserNode basics, offload to a **Web Worker** to keep the main thread free for rendering. **AudioWorklet** (the modern replacement for the deprecated ScriptProcessorNode) runs custom processing on a dedicated audio thread with guaranteed timing — ideal for onset detection that needs precise sample-level accuracy. AudioWorklet requires a secure context (HTTPS or localhost).

### How LedFx structures its pipeline

LedFx — the most mature open-source music-to-LED system — provides a reference architecture. Its core singleton orchestrates subsystems via asyncio. A **registry pattern** with factory methods dynamically discovers Devices, Virtuals, Effects, and Integrations. Audio capture runs at ~60 fps via the `sounddevice` library. A pre-emphasis filter and Phase Vocoder FFT (via an aubio fork) feed 3-resolution **Mel filterbanks** for frequency features. Effects inherit from base classes and are auto-registered. The "Virtual" abstraction decouples effect rendering from physical device topology — a logical LED strip spans multiple physical devices through segment mapping. Output distributes pixel data via UDP (DDP, E1.31/sACN, ArtNet) and other protocols. LedFx v3 adds Spotify API integration, enabling anticipatory effects by reading pre-analyzed beat/section data.

WLED Sound Reactive takes a contrasting approach — embedded firmware on ESP32 microcontrollers with on-device FFT. One device can broadcast audio analysis data via UDP multicast to all other WLED devices, which render effects locally. This eliminates network latency for audio but limits processing power.

The **common patterns across all projects**: FFT as universal foundation, frequency band splitting (3–16 bands), temporal smoothing to prevent jitter, auto-gain/normalization, network distribution of output data (UDP is universal), and modular effect architecture with plugin/registry patterns.

---

## Conclusion: building a complete system

The path from microphone input to synchronized Hue bulbs requires careful orchestration of DSP, beat tracking, perceptual science, and protocol engineering. The most critical insight is that **the system must be designed around the Hue Entertainment API's ~12.5 fps effective rate and ~80–120ms end-to-end latency** — not against them. Use the bridge's built-in interpolation for smooth transitions. Use beat prediction (PLL) to trigger commands early, compensating for pipeline latency. Use the hybrid reactive-generative approach so lights always look good regardless of what the music is doing.

Three numbers define the design space: **25 Hz** (bridge streaming rate), **3 Hz** (maximum safe flash rate), and **~80ms** (target total latency). An FFT size of 2048 with Hann windowing at 44.1 kHz provides the spectral foundation. Asymmetric EMA smoothing (fast attack, slow release) with logarithmic brightness mapping produces the most musical-feeling light behavior. Energy-based bass detection with autocorrelation BPM estimation and PLL tracking handles the rhythmic engine. The spectral centroid driving hue position, modulated by section-level intensity adaptation, creates the color story.

The architectural lesson from LedFx, WLED, and commercial products like iLightShow is consistent: keep audio analysis running at high frame rates internally (60+ fps), decouple it from the output rate, and let a separate transmission thread manage the bridge's slower cadence with peak-hold logic to preserve transient impacts. For the web platform specifically, the AnalyserNode handles FFT efficiently, Web Workers offload heavy computation, and a local WebSocket-to-DTLS proxy bridges the browser's inability to send raw UDP to the Hue Entertainment API. This architecture is both practical to implement and capable of producing genuinely immersive music-reactive lighting.
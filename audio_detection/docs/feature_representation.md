# Feature Representation

## Acoustic Properties of Rotary-Wing UAVs

Rotary-wing drones produce characteristic acoustic signatures:

- **Fundamental frequency:** Tied to rotor RPM. For consumer drones, typically 80–400 Hz.
- **Harmonics:** Integer multiples of the fundamental (2f, 3f, 4f...). The harmonic structure is the most distinctive feature.
- **Blade passage frequency:** Additional tonal component at N × RPM/60, where N = number of blades.
- **Amplitude modulation:** Due to blade rotation; produces sidebands around harmonics.
- **Broadband noise:** From turbulence over blade surfaces.

The key insight: drone audio is **tonal and harmonic** with relatively stable frequency content over short windows. The detection problem is about finding this structured, periodic signal within arbitrary background noise.

---

## Default: Log-Mel Spectrogram

**Configuration:**
```yaml
n_fft: 1024        # FFT size (~64ms at 16kHz)
hop_length: 160    # 10ms frame shift
n_mels: 128        # mel filter banks
f_min: 20.0        # Hz
f_max: 8000.0      # Hz
top_db: 80.0       # dynamic range
```

**For a 1-second window at 16kHz:**
- Input: 16,000 samples
- Output: (128, 101) — 128 mel bins × 101 time frames
- Feature map size: ~13k values

**Why log-mel works:**
1. Logarithmic frequency axis aligns with harmonic spacing (harmonics appear equally spaced)
2. Logarithmic amplitude (dB) compresses the dynamic range
3. Mel scale is perceptually motivated but also practically useful for drone detection
4. Computationally efficient — GPU-accelerated via torchaudio

---

## Alternative: CQT (Constant-Q Transform)

Available via `feature_type: cqt` in config.

**Advantages over log-mel:**
- Constant frequency resolution ratio (Q = f/Δf) — genuinely logarithmic, not approximated
- Better suited for harmonic analysis
- Can resolve closely-spaced harmonics better at low frequencies

**Disadvantages:**
- Slower to compute (no FFT shortcut for arbitrary Q)
- Less well-studied in neural audio classification benchmarks

**When to try it:** If log-mel models fail to separate drone from tonal background noise (e.g., machinery, HVAC), the better harmonic resolution of CQT may help.

---

## Rejected: MFCC

MFCCs are derived from mel spectrograms by applying a Discrete Cosine Transform and retaining the lower-order coefficients. This process:

1. Discards higher-frequency spectral detail
2. Explicitly decorrelates frequency bands
3. **Discards pitch information** (the cepstral liftering step removes the fundamental period)

MFCC was designed for speech recognition, where pitch is a nuisance variable (the same word sounds the same at different pitches). For drone detection, pitch IS the signal — different drones at different RPMs have different fundamental frequencies. MFCCs are the wrong representation for this task.

---

## Normalization

Feature maps are not globally normalized before being passed to the model. Instead:
- The log-mel operation bounds values to `[−top_db, 0]` dB
- The model's first batch normalization layer (if present) handles input distribution normalization

If a global normalization scheme is needed (e.g., for architectures without batch norm), compute mean and std from the training set and store them with the model checkpoint.

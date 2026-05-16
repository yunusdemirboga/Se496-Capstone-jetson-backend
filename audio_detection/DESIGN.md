# System Design — UAV Audio Detection

This document records the architecture decisions, their rationale, and the alternatives that were considered and rejected. Read this before modifying any core pipeline component.

---

## 1. Problem Framing

**Task:** Given a fixed-duration audio window, does it contain acoustic energy consistent with rotary-wing UAV operation?

This framing is deliberate. The naive framing — "classify a clip as drone or non-drone" — is fragile because clip length encodes the label in the DADS dataset (drone clips are short, non-drone clips are long). By committing to fixed-length windows, the model must classify based on spectro-temporal content.

**Deployment scenario:** Sliding-window detection over a continuous audio stream. The model sees 1-second windows and emits a binary decision (or probability) for each.

---

## 2. Dataset Problems and Mitigations

### 2.1 Source Leakage (Root Cause of False High Accuracy)

**Problem:** The DADS dataset merges 10 source datasets with no explicit `source_id` column. A random 80/20 split will place clips from the same recording session in both training and test, allowing the model to memorize recording-condition fingerprints rather than generalizable drone acoustics.

**Fix:** Source-aware clustering-based splits. We cluster clips by acoustic fingerprint (spectral centroid, bandwidth, rolloff, RMS, duration) and assign entire clusters to a single split. See `src/data/splitting.py`.

**Alternative considered:** Using clip filename metadata to identify sources. Rejected because the HuggingFace parquet conversion does not preserve original filenames.

### 2.2 Duration Leakage

**Problem:** Drone clips average 0.60s; non-drone clips average 7.28s. Any model processing variable-length inputs can exploit clip duration as a trivial discriminator.

**Fix:** Fixed 1-second sliding windows applied uniformly to all clips. Short clips are zero-padded; long clips yield multiple windows. All windows are the same length before feature extraction.

**Window size choice:** 1 second. Rationale: short enough to be computationally tractable and compatible with the shortest meaningful drone acoustic signature; long enough to contain multiple rotor rotation cycles for most consumer drones (rotor period ≈ 3–15ms).

### 2.3 Class Imbalance (9:1 drone:non-drone by clip count)

**Problem:** A model predicting "drone" for everything achieves ~90% accuracy.

**Fix (primary):** The windowing step naturally corrects this. Non-drone clips are long (avg 7.28s), so a 1-second window with 0.5s stride yields ~13 windows per clip. Drone clips are short (avg 0.60s), yielding ~1 window per clip. Post-windowing balance is approximately 163k drone windows vs 216k non-drone windows — close to 1:1.4.

**Fix (secondary):** Class-weighted cross-entropy loss for any residual imbalance. Weights are computed from the actual training split, not assumed.

### 2.4 Low Drone Diversity

**Problem:** Drone clips in DADS cluster acoustically — they appear to come from constrained recording conditions with limited variation in drone model, environment, and microphone placement.

**Fix:** Stochastic augmentation during training. Background mixing (the primary intervention) forces the model to separate drone acoustics from arbitrary background sounds. See Section 4.

---

## 3. Feature Representation

**Decision: Log-mel spectrogram as default.**

Drone acoustics are tonal and harmonic (fundamental + harmonics of rotor RPM, typically 80–400 Hz for consumer drones). The log-mel spectrogram:

- Preserves tonal structure across the relevant frequency range
- Uses logarithmic frequency axis (perceptually aligned; harmonics appear at equal intervals)
- Is computationally efficient
- Has strong precedent in audio classification (AudioSet, ESC-50 benchmarks)

**Configuration:** 128 mel bins, FFT size 1024 (~64ms at 16kHz), hop length 160 samples (10ms), f_min=20Hz, f_max=8000Hz.

**Rejected: MFCC.** MFCCs are designed for speech and are explicitly constructed to discard pitch information via the cepstral liftering step. Pitch is a key discriminator for drone detection. MFCCs are the wrong prior.

**Available alternative: CQT.** Constant-Q Transform offers better logarithmic frequency resolution and is naturally suited for harmonic analysis. It is available via `feature_type: cqt` in the config. Benchmarked against log-mel in Phase 3.

**Raw waveform:** Not used as primary input. Would require the model to learn all frequency-domain invariances from scratch, increasing data requirements significantly.

---

## 4. Augmentation Strategy

Augmentation is applied **stochastically during training** (not pre-computed). Each training epoch sees different augmented versions of each clip. This is implicit regularization.

### 4.1 Background Mixing (Primary)

Mix drone windows with real environmental sounds at varying SNR levels.

```
mixed = drone + scale * background
where scale is chosen to achieve target SNR ∈ [−5, 20] dB
```

Applied only to drone (label=1) clips. Rationale: a drone clip mixed with background noise still contains drone audio — the label is preserved. A non-drone clip mixed with another non-drone sound is also unambiguous. We do not mix non-drone clips with drone audio (that would require changing the label).

SNR range of −5 to +20 dB means the model sees cases where the background is louder than the drone. This directly targets the failure mode observed in real-world testing.

**Background sound library:** Located in `data/backgrounds/`. Should contain diverse real-world environmental sounds. ESC-50, UrbanSound8K subsets, and DEMAND noise recordings are suitable. These files are downloaded separately and are not part of the train/val/test dataset.

### 4.2 Additive Noise

Gaussian and pink noise at SNR 10–30 dB. Models microphone noise floor and electrical interference.

### 4.3 Pitch Shift (±2 semitones)

Simulates different drone models operating at different RPMs, or Doppler shift from drone movement.

### 4.4 Time Stretch (0.9×–1.1×)

Subtle temporal variation. Output is trimmed/padded to preserve fixed length.

---

## 5. Split Strategy

Source-based clustering via agglomerative clustering (Ward linkage) on acoustic fingerprints.

**Process:**
1. For each clip, compute: log(duration), spectral centroid (mean), spectral bandwidth (mean), spectral rolloff (mean), RMS energy (mean)
2. Cluster within each class separately (drone clusters and non-drone clusters independently)
3. Assign entire clusters to train (70%) / val (15%) / test (15%)
4. Propagate cluster assignments to all windows derived from each clip

**Why separate clustering per class:** Drone and non-drone clips live in different regions of acoustic feature space. Joint clustering would produce clusters that are dominated by one class and would not meaningfully separate recording sessions.

**Determinism:** Splits are generated once, saved as CSV manifests in `data/splits/`, and committed to version control. The split generation script is not run again unless explicitly needed (with a new seed and documented reason).

---

## 6. Evaluation Protocol

See `docs/evaluation_protocol.md` for the full protocol. Key principle: **never report a single accuracy number**.

Four evaluation tiers are defined. The primary optimization target is **Tier 2 (out-of-distribution)**, not Tier 1 (in-distribution).

Primary reported metric: **F1 score on Tier 2**, with PR-AUC as secondary. We also report TPR at FPR=5% as a deployment-relevant operating point.

---

## 7. Framework

**PyTorch + torchaudio.**

Rationale:
- torchaudio provides GPU-accelerated mel spectrogram computation
- The HuggingFace `datasets` library has native PyTorch tensor output
- Most audio ML research uses PyTorch; examples and pretrained models are widely available
- Flexible enough for custom model architectures without boilerplate overhead

---

## 8. What This System Does Not Do

- It does not perform localization (does not tell you where in a longer recording the drone is)
- It does not identify drone type or model
- It has not been tested on fixed-wing UAVs (different acoustic signature)
- It does not operate in real-time without a sliding-window inference wrapper
- Performance below −5 dB SNR is not guaranteed

See `docs/known_limitations.md` for the full list.

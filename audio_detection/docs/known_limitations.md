# Known Limitations

This document records what the system cannot do, what has not been tested, and what assumptions underlie the current design. Update this document as limitations are discovered or resolved.

---

## Acoustic Scope

**Fixed-wing UAVs are not supported.**
Fixed-wing UAVs produce different acoustic signatures (propeller noise at higher frequencies, less tonal harmonic structure). The training data contains only rotary-wing drones. Performance on fixed-wing UAVs is unknown and likely poor.

**Propeller aircraft may produce false positives.**
Some propeller-driven aircraft produce acoustic signatures similar to multi-rotor drones. This is a fundamental ambiguity in audio-only detection.

**High-altitude drones may not be detectable.**
The training data includes some distant-microphone recordings, but performance at low drone SNR (< −10 dB) is not guaranteed. See the SNR sweep results in evaluation reports.

---

## Environmental Conditions

**Not tested in high-wind conditions.**
Strong wind generates broadband low-frequency noise that may mask drone harmonics and/or trigger false positives.

**Not tested in rain.**
Rain produces broadband noise and may attenuate higher drone harmonics.

**High-urban-noise environments may increase false positives.**
Machinery (HVAC, motors, generators) produces tonal harmonic noise that can resemble drone signatures at certain RPMs.

---

## Dataset Assumptions

**Source grouping is approximate.**
The DADS dataset has no `source_id` column. Our source-based splits use acoustic clustering as a proxy, which may not perfectly separate all recording sessions. Some leakage between train and test may remain.

**Background library quality affects augmentation quality.**
The robustness of the trained model is partly a function of how diverse the `data/backgrounds/` library is. A narrow background library will produce a model that is only robust to the types of noise it has seen.

---

## Operational Limitations

**Not a real-time system.**
The current implementation processes pre-recorded audio. A sliding-window inference wrapper is needed for streaming operation. See `docs/deployment_notes.md` (to be written).

**Single-channel (mono) audio only.**
The model processes mono audio. Stereo or multichannel inputs must be downmixed before inference.

**Does not localize the drone.**
The system outputs a binary classification per window. It does not tell you where in space the drone is, or where in a longer recording the drone appears (though a sliding-window scan can approximate this).

**Does not identify drone type or model.**
Binary classification only: drone present or not present.

---

## Performance Claims

No performance numbers should be quoted from Tier 1 (in-distribution) evaluation as evidence of deployment readiness. Only Tier 2 (OOD) and Tier 4 (real-world) numbers are relevant for that claim.

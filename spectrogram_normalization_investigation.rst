Spectrogram normalization for EfficientNet
===========================================

Status
------

Investigation complete. Do not enable per-sample standardization by default.
Retain the current ``normalize: false`` default until a training-split
normalization experiment demonstrates an improvement for the target task.

Evidence
--------

TorchSIG's ``compute_spectrogram`` returns power in dB. It floors each sample
100 dB below that sample's peak, but it does not normalize FFT length or remove
the sample's absolute dB offset. Training and inference currently pass the same
``normalize`` setting into the model, and all supplied spectrogram EfficientNet
parameter files set it to ``false``.

The optional ``SpectrogramNormalization`` computes a mean and standard
deviation independently for every sample and channel. Consequently, it is
invariant to an affine offset and scale within each spectrogram. In particular,
scaling an IQ example by 0.1 shifts its power spectrogram by approximately
-20 dB, while its per-sample standardized tensor is effectively unchanged.
This removes absolute received-power and noise-floor cues. It still preserves
within-sample structure, including signal-to-noise contrast, so it may remain a
useful opt-in for tasks where absolute power is a nuisance variable.

The deterministic diagnostic in
``examples/investigate_spectrogram_normalization.py`` demonstrates this result.
It also shows that one mean and standard deviation fitted on training data
stabilize the input distribution without making power-shifted samples
identical. Run it with::

    python examples/investigate_spectrogram_normalization.py

Recommendation
--------------

Use the unnormalized dB spectrogram as the baseline. Evaluate training-dataset
standardization as the leading alternative: estimate one mean and standard
deviation per input channel from the training split only, freeze those values,
and apply the same affine transform to training, validation, test, and
standalone inference. Do not estimate statistics independently on validation
or test data.

Dataset-level statistics are preferable to per-sample statistics when received
power, noise-floor level, or SNR may be predictive or operationally relevant.
Per-sample normalization should remain explicitly selectable for tasks that
need invariance to those quantities. No universal default should be selected
without task-level measurements.

Implementation plan
-------------------

1. Add a streaming, float64 training-split statistics estimator so the full
   dataset is never held in memory. Store sample count, per-channel mean, and
   per-channel standard deviation.
2. Add an explicit normalization mode rather than extending the Boolean:
   ``none``, ``per_sample``, or ``dataset``. Preserve ``normalize`` as a
   compatibility alias during migration.
3. Require frozen mean and standard deviation for ``dataset`` mode. Save them
   in the checkpoint hyperparameters and training result metadata, and fail
   clearly during inference if they are missing.
4. Apply normalization in the model wrapper so CPU/GPU and training/inference
   behavior remain identical. Register fitted statistics as buffers.
5. Compare all three modes with identical seeded splits. Report overall and
   per-SNR accuracy/F1, calibration, convergence, and performance under
   shifted noise power. Repeat across several seeds.
6. Promote dataset normalization to the default only if it improves aggregate
   performance without unacceptable degradation in SNR strata or shifted-noise
   evaluation. Otherwise retain ``none`` and document task-specific guidance.

Experimental controls
---------------------

Use one FFT size within each experiment because TorchSIG spectrogram levels are
not comparable across FFT sizes. Fit statistics after the exact spectrogram
transform used for training. Keep dataset generation seeds, model
initialization, optimizer settings, and checkpoint selection fixed across
normalization modes. Record statistics and configuration alongside every run
so inference cannot silently use a different transform.

# Model card — v0.1.0

Research code; no real-world checkpoint is distributed. The demonstration injects artificial signal and cannot establish domain validity.

Inputs: one preselected observation per sample, optional paired 8-bit RGB and calibrated spectral cube. Outputs: class probability, F/M/REVIEW, model version, research-only flag.

Algorithms: fixed statistical features, training-only standardization, RBF-SVM selected on validation AUC, separate sigmoid calibration, validation-selected abstention threshold. Fusion concatenates features; it is not a neural network.

Scope: one documented device/protocol and declared acquisition interval. No live camera integration or mechanical action. Invalid metadata, poor signal, incompatible wavelengths, or insufficient validation evidence lead to REVIEW.

Risks/limitations: informative background and batch artifacts, unreliable labels, survivor bias, limited independent samples, domain shift, correlated batches, uneven rejection rates. Current intervals assume independent samples. No full out-of-distribution detector is implemented.

Before deployment: blinded prospective batch validation, per-class error and coverage assessment, device-specific reference validation, complete-cycle timing, external-domain evaluation, and verified model export if required. Exact performance targets must be specified for the intended process.

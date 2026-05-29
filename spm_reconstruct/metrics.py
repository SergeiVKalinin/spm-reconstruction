"""
spm_eval/metrics.py
-------------------
Pure metric functions for evaluating SPM reconstructions.

Every function takes numpy arrays and returns a scalar or array.
No classes, no side effects, no imports from spm_reconstruct.

Functions
---------
Accuracy (require true_surface):
    rmse(predicted, true)
    mae(predicted, true)
    rmse_by_line(predicted, true)
    rmse_by_region(predicted, true, mask, threshold)

Structural:
    ssim(predicted, true)

Spectral:
    psd_rmse(predicted, true)

Uncertainty calibration (require uncertainty):
    coverage(predicted, uncertainty, true, level)
    mean_uncertainty(uncertainty)
    nll(predicted, uncertainty, true)

Classifier quality (require quality map + artifact_mask):
    classifier_auc(quality, artifact_mask)
    classifier_fpr(quality, artifact_mask, threshold)
    classifier_fnr(quality, artifact_mask, threshold)
"""

import numpy as np


# ── Accuracy ──────────────────────────────────────────────────────────────────

def rmse(predicted, true):
    """Root mean squared error over the full image."""
    return float(np.sqrt(np.mean((predicted - true) ** 2)))


def mae(predicted, true):
    """Mean absolute error over the full image."""
    return float(np.mean(np.abs(predicted - true)))


def rmse_by_line(predicted, true):
    """
    Per-line RMSE.

    Returns
    -------
    np.ndarray of shape (Ny,)
    """
    return np.sqrt(np.mean((predicted - true) ** 2, axis=1))


def rmse_by_region(predicted, true, mask, threshold=0.5):
    """
    RMSE restricted to pixels where mask > threshold.

    Used to compute RMSE separately for:
      - corrupted region : mask = artifact_mask,  threshold = 0.5
      - clean region     : mask = 1-artifact_mask, threshold = 0.5

    Returns
    -------
    float or None if no pixels pass the threshold
    """
    px = mask > threshold
    if px.sum() == 0:
        return None
    return float(np.sqrt(np.mean((predicted[px] - true[px]) ** 2)))


# ── Structural ────────────────────────────────────────────────────────────────

def ssim(predicted, true, data_range=1.0):
    """
    Structural Similarity Index (SSIM).

    Measures perceptual similarity rather than pixel-level error.
    Range: [-1, 1], higher is better. Perfect reconstruction = 1.0.

    Parameters
    ----------
    data_range : float — range of the data (default 1.0 for normalised images)
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_p = predicted.mean()
    mu_t = true.mean()
    sig_p = predicted.std()
    sig_t = true.std()
    sig_pt = np.mean((predicted - mu_p) * (true - mu_t))

    numerator   = (2*mu_p*mu_t + c1) * (2*sig_pt + c2)
    denominator = (mu_p**2 + mu_t**2 + c1) * (sig_p**2 + sig_t**2 + c2)
    return float(numerator / denominator)


# ── Spectral ──────────────────────────────────────────────────────────────────

def psd_rmse(predicted, true):
    """
    Relative power spectral density error along the fast-scan axis.

    Measures how well the reconstruction recovers each spatial frequency.
    Low at all frequencies = no over-smoothing, no artifact ringing.

    Returns a single scalar: RMS of the relative PSD error.
    """
    p_psd = np.mean(
        [np.abs(np.fft.rfft(predicted[k])) ** 2
         for k in range(predicted.shape[0])], axis=0)
    t_psd = np.mean(
        [np.abs(np.fft.rfft(true[k])) ** 2
         for k in range(true.shape[0])], axis=0)

    # Relative error (avoid division by zero at DC)
    rel_err = np.abs(p_psd - t_psd) / (t_psd + 1e-10)
    return float(np.sqrt(np.mean(rel_err ** 2)))


def psd_curves(predicted, true):
    """
    Return (freqs, psd_predicted, psd_true) for plotting.

    Useful for diagnosing over-smoothing: if psd_predicted drops
    below psd_true at mid-to-high frequencies, the algorithm is
    removing genuine surface features.
    """
    Nx = predicted.shape[1]
    freqs = np.fft.rfftfreq(Nx)
    p_psd = np.mean(
        [np.abs(np.fft.rfft(predicted[k])) ** 2
         for k in range(predicted.shape[0])], axis=0)
    t_psd = np.mean(
        [np.abs(np.fft.rfft(true[k])) ** 2
         for k in range(true.shape[0])], axis=0)
    return freqs, p_psd, t_psd


# ── Uncertainty calibration ───────────────────────────────────────────────────

def coverage(predicted, uncertainty, true, level=0.95):
    """
    Fraction of true errors that fall within level * uncertainty.

    A well-calibrated algorithm should return ~level here.
    E.g. coverage(..., level=0.95) should be close to 0.95.

    Values significantly below level: uncertainty is underestimated
    (algorithm is overconfident).
    Values significantly above level: uncertainty is overestimated
    (algorithm is too conservative).
    """
    # For a Gaussian: P(|error| < z*sigma) = level
    # => z = sqrt(2) * erfinv(level)
    from scipy.special import erfinv
    z   = float(np.sqrt(2) * erfinv(level))
    err = np.abs(predicted - true)
    return float(np.mean(err <= z * uncertainty))


def mean_uncertainty(uncertainty):
    """Mean posterior uncertainty (sharpness). Lower = more confident."""
    return float(uncertainty.mean())


def nll(predicted, uncertainty, true):
    """
    Mean negative log-likelihood under a Gaussian predictive distribution.

    Treats (predicted, uncertainty) as a Gaussian predictive distribution
    and evaluates it on the true surface. Lower is better.

    NLL = 0.5 * mean[ log(2*pi*sigma^2) + (y - mu)^2 / sigma^2 ]
    """
    sigma2  = np.maximum(uncertainty ** 2, 1e-10)
    err2    = (predicted - true) ** 2
    per_px  = 0.5 * (np.log(2 * np.pi * sigma2) + err2 / sigma2)
    return float(per_px.mean())


# ── Classifier quality ────────────────────────────────────────────────────────

def classifier_auc(quality, artifact_mask, mask_threshold=0.5):
    """
    AUC of the quality map as a binary artifact classifier.

    Treats (1 - quality) as a score for being corrupted, and
    (artifact_mask > mask_threshold) as the true binary label.

    Higher AUC = better discrimination between clean and corrupted pixels.
    AUC = 0.5 means random; AUC = 1.0 means perfect.

    Parameters
    ----------
    mask_threshold : float — artifact_mask threshold for binary label
    """
    y_true  = (artifact_mask > mask_threshold).ravel().astype(int)
    y_score = (1.0 - quality).ravel()

    # Compute AUC via trapezoid rule on ROC curve
    order   = np.argsort(y_score)[::-1]
    y_sort  = y_true[order]
    tp = np.cumsum(y_sort)
    fp = np.cumsum(1 - y_sort)
    tpr = tp / (tp[-1] + 1e-10)
    fpr = fp / (fp[-1] + 1e-10)
    # Prepend (0,0) for proper trapezoid
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapezoid(tpr, fpr))


def classifier_fpr(quality, artifact_mask,
                   quality_threshold=0.5, mask_threshold=0.5):
    """
    False positive rate at a given quality threshold.

    FPR = fraction of clean pixels incorrectly flagged as corrupted.
    High FPR causes over-smoothing: clean pixels are down-weighted.

    Parameters
    ----------
    quality_threshold : float — pixels with quality < threshold flagged as corrupted
    mask_threshold    : float — artifact_mask threshold for true label
    """
    true_clean   = artifact_mask <= mask_threshold
    pred_corrupt = quality < quality_threshold
    fp = np.sum(pred_corrupt & true_clean)
    tn = np.sum(~pred_corrupt & true_clean)
    return float(fp / (fp + tn + 1e-10))


def classifier_fnr(quality, artifact_mask,
                   quality_threshold=0.5, mask_threshold=0.5):
    """
    False negative rate at a given quality threshold.

    FNR = fraction of corrupted pixels not detected.
    High FNR allows artifact residuals to remain in the reconstruction.

    Parameters
    ----------
    quality_threshold : float — pixels with quality < threshold flagged as corrupted
    mask_threshold    : float — artifact_mask threshold for true label
    """
    true_corrupt = artifact_mask > mask_threshold
    pred_corrupt = quality < quality_threshold
    fn = np.sum(~pred_corrupt & true_corrupt)
    tp = np.sum(pred_corrupt & true_corrupt)
    return float(fn / (fn + tp + 1e-10))

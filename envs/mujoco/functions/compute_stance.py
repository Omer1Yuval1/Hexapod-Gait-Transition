import numpy as np
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

def compute_stance(
    x_ij: np.ndarray,
    # min_peak_prominence: float,
    direction: int,
) -> np.ndarray:
    """
    Given a 1D signal x_ij, returns a binary 'stance' vector (1 = stance, 0 = swing).
    
    Parameters
    ----------
    x_ij : np.ndarray
        The signal for one leg (shape: [n_samples,]).
    min_peak_prominence : float
        Minimum prominence for peak-finding.
    direction : Int
        1 for forward gait, -1 for backward gait.
    
    Returns
    -------
    stance : np.ndarray
        A 0/1 array of the same shape as x_ij.
    """

    min_peak_prominence = np.std(x_ij) # 0.025

    # print("Standard Deviation of x_ij:", np.round(np.std(x_ij),3))

    # find all local maxima and minima
    pos_peaks, _ = find_peaks(x_ij, prominence=min_peak_prominence)
    neg_peaks, _ = find_peaks(-x_ij, prominence=min_peak_prominence)
    
    stance = np.ones_like(x_ij, dtype=int)
    
    if direction == -1: # backward gait.
        if pos_peaks.size > 0 and ((neg_peaks.size > 0 and pos_peaks[0] < neg_peaks[0]) or neg_peaks.size == 0):
            stance[: pos_peaks[0] + 1] = 0
        
        # 2) for each trough, zero from trough → next peak (or to end)
        for trough_idx in neg_peaks:
            # find the very next positive peak
            later_pos = pos_peaks[pos_peaks > trough_idx]
            if later_pos.size > 0:
                end_idx = later_pos[0]
            else:
                end_idx = len(x_ij) - 1
            stance[trough_idx : end_idx + 1] = 0
    
    else: # forward gait

        if neg_peaks.size > 0 and ((pos_peaks.size > 0 and neg_peaks[0] < pos_peaks[0]) or pos_peaks.size == 0):
            stance[: neg_peaks[0] + 1] = 0
        
        # 2) for each positive peak, zero from peak → next trough (or to end)
        for peak_idx in pos_peaks:
            later_neg = neg_peaks[neg_peaks > peak_idx]
            if later_neg.size > 0:
                end_idx = later_neg[0]
            else:
                end_idx = len(x_ij) - 1
            stance[peak_idx : end_idx + 1] = 0

    if(False):
        # Plot the signal with peaks
        plt.figure(figsize=(10, 6))
        plt.plot(x_ij, label="Signal")
        plt.plot(pos_peaks, x_ij[pos_peaks], "ro", label="Positive Peaks")
        plt.plot(neg_peaks, x_ij[neg_peaks], "bo", label="Negative Peaks")

        # Color the graph in the ranges corresponding to zero in the stance array
        for i in range(len(stance)):
            if stance[i] == 0:
                plt.axvspan(i - 0.5, i, color='red', alpha=0.3)
            else:
                plt.axvspan(i, i + 0.5, color='green', alpha=0.3)

        plt.title("Signal with Peaks")
        plt.xlabel("Sample Index")
        plt.ylabel("Signal Value")
        plt.legend()
        plt.grid()
        plt.show()

    return stance

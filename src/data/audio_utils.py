# SPAPL 2024

import torch
import numpy as np
from typing import List, Optional, Union

from transformers.audio_utils import hertz_to_mel, mel_to_hertz
# my mel filter with the VTLN warping 

def _create_triangular_filter_bank(fft_freqs: np.ndarray, 
    filter_freqs: np.ndarray, 
    vtln_low : float, 
    vtln_high: float, 
    vtln_warp_factor: float,
    triangularize_in_mel_space: bool = False,
    ) -> np.ndarray:
    """
    Creates a triangular filter bank.

    Adapted from *torchaudio* and *librosa*.

    Args:
        fft_freqs (`np.ndarray` of shape `(num_frequency_bins,)`):
            Discrete frequencies of the FFT bins in Hz.
        filter_freqs (`np.ndarray` of shape `(num_mel_filters,)`):
            Center frequencies of the triangular filters to create, in Hz.

    Returns:
        `np.ndarray` of shape `(num_frequency_bins, num_mel_filters)`
    """
    
    if vtln_warp_factor != 1.0:
        low_freq, high_freq = filter_freqs[0], filter_freqs[-1]
        if triangularize_in_mel_space:
            filter_freqs = hertz_to_mel(vtln_warp_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, mel_to_hertz(filter_freqs)))
        else:
            filter_freqs = vtln_warp_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, filter_freqs)
    
    filter_diff = np.diff(filter_freqs)

    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]

    if vtln_warp_factor == 1.0:
        bins = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    else:
        # warping can move the order of left_mel, center_mel, right_mel anywhere
        bins = np.zeros_like(down_slopes)
        fft_freqs = np.expand_dims(fft_freqs, 1)
        filter_freq = np.expand_dims(filter_freqs, 0)
        left_freq = filter_freq[:, :-2]
        center_freq = filter_freq[:, 1:-1]
        right_freq = filter_freq[:, 2:]
        down_idx = np.greater(fft_freqs, left_freq) & np.less_equal(fft_freqs, center_freq)  # left_mel < mel <= center_mel
        up_idx = np.greater(fft_freqs, center_freq) & np.less(fft_freqs, right_freq)  # center_mel < mel < right_mel
        bins[down_idx] = down_slopes[down_idx]
        bins[up_idx] = up_slopes[up_idx]
    return bins, filter_freqs

def vtln_warp_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq: float,
    high_freq: float,
    vtln_warp_factor: float,
    freq: np.ndarray,
) -> np.ndarray:
    r"""This computes a VTLN warping function that is not the same as HTK's one,
    but has similar inputs (this function has the advantage of never producing
    empty bins).

    This function computes a warp function F(freq), defined between low_freq
    and high_freq inclusive, with the following properties:
        F(low_freq) == low_freq
        F(high_freq) == high_freq
    The function is continuous and piecewise linear with two inflection
        points.
    The lower inflection point (measured in terms of the unwarped
        frequency) is at frequency l, determined as described below.
    The higher inflection point is at a frequency h, determined as
        described below.
    If l <= f <= h, then F(f) = f/vtln_warp_factor.
    If the higher inflection point (measured in terms of the unwarped
        frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
        Since (by the last point) F(h) == h/vtln_warp_factor, then
        max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
        h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
          = vtln_high_cutoff * min(1, vtln_warp_factor).
    If the lower inflection point (measured in terms of the unwarped
        frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
        This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
                            = vtln_low_cutoff * max(1, vtln_warp_factor)
    Args:
        vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
        vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
        low_freq (float): Lower frequency cutoffs in mel computation
        high_freq (float): Upper frequency cutoffs in mel computation
        vtln_warp_factor (float): Vtln warp factor
        freq (Tensor): given frequency in Hz

    Returns:
        Tensor: Freq after vtln warp
    """
    assert vtln_low_cutoff > low_freq, "be sure to set the vtln_low option higher than low_freq"
    assert vtln_high_cutoff < high_freq, "be sure to set the vtln_high option lower than high_freq [or negative]"
    l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l  # F(l)
    Fh = scale * h  # F(h)
    assert l > low_freq and h < high_freq
    # slope of left part of the 3-piece linear function
    scale_left = (Fl - low_freq) / (l - low_freq)
    # [slope of center part is just "scale"]

    # slope of right part of the 3-piece linear function
    scale_right = (high_freq - Fh) / (high_freq - h)

    res = np.empty_like(freq)

    outside_low_high_freq = np.less(freq, low_freq) | np.greater(freq, high_freq)  # freq < low_freq || freq > high_freq
    before_l = np.less(freq, l)  # freq < l
    before_h = np.less(freq, h)  # freq < h
    after_h = np.greater_equal(freq, h)  # freq >= h

    # order of operations matter here (since there is overlapping frequency regions)
    res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
    res[before_h] = scale * freq[before_h]
    res[before_l] = low_freq + scale_left * (freq[before_l] - low_freq)
    res[outside_low_high_freq] = freq[outside_low_high_freq]

    return res


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
    vtln_warp_factor: float = 1.0,
    vtln_low: float = 20.0,
    vtln_high: float = -500.0,
    triangularize_in_mel_space: bool = False,
) -> np.ndarray:
    """
    Creates a frequency bin conversion matrix used to obtain a mel spectrogram. This is called a *mel filter bank*, and
    various implementation exist, which differ in the number of filters, the shape of the filters, the way the filters
    are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The goal of these
    features is to approximate the non-linear human perception of the variation in pitch with respect to the frequency.

    Different banks of mel filters were introduced in the literature. The following variations are supported:

    - MFCC FB-20: introduced in 1980 by Davis and Mermelstein, it assumes a sampling frequency of 10 kHz and a speech
      bandwidth of `[0, 4600]` Hz.
    - MFCC FB-24 HTK: from the Cambridge HMM Toolkit (HTK) (1995) uses a filter bank of 24 filters for a speech
      bandwidth of `[0, 8000]` Hz. This assumes sampling rate ≥ 16 kHz.
    - MFCC FB-40: from the Auditory Toolbox for MATLAB written by Slaney in 1998, assumes a sampling rate of 16 kHz and
      speech bandwidth of `[133, 6854]` Hz. This version also includes area normalization.
    - HFCC-E FB-29 (Human Factor Cepstral Coefficients) of Skowronski and Harris (2004), assumes a sampling rate of
      12.5 kHz and speech bandwidth of `[0, 6250]` Hz.

    This code is adapted from *torchaudio* and *librosa*. Note that the default parameters of torchaudio's
    `melscale_fbanks` implement the `"htk"` filters while librosa uses the `"slaney"` implementation.

    Args:
        num_frequency_bins (`int`):
            Number of frequencies used to compute the spectrogram (should be the same as in `stft`).
        num_mel_filters (`int`):
            Number of mel filters to generate.
        min_frequency (`float`):
            Lowest frequency of interest in Hz.
        max_frequency (`float`):
            Highest frequency of interest in Hz. This should not exceed `sampling_rate / 2`.
        sampling_rate (`int`):
            Sample rate of the audio waveform.
        norm (`str`, *optional*):
            If `"slaney"`, divide the triangular mel weights by the width of the mel band (area normalization).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.
        triangularize_in_mel_space (`bool`, *optional*, defaults to `False`):
            If this option is enabled, the triangular filter is applied in mel space rather than frequency space. This
            should be set to `true` in order to get the same results as `torchaudio` when computing mel filters.

    Returns:
        `np.ndarray` of shape (`num_frequency_bins`, `num_mel_filters`): Triangular filter bank matrix. This is a
        projection matrix to go from a spectrogram to a mel spectrogram.
    """
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    if triangularize_in_mel_space:
        # frequencies of FFT bins in Hz, but filters triangularized in mel space
        fft_bin_width = sampling_rate / (num_frequency_bins * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        # frequencies of FFT bins in Hz
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    mel_filters, filter_freqs = _create_triangular_filter_bank(fft_freqs, filter_freqs, vtln_low, vtln_high, vtln_warp_factor, triangularize_in_mel_space)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        if vtln_warp_factor == 1.0:
            enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        else:
            filter_diff = (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
            values, counts = np.unique(filter_diff, return_counts=True)
            common_value = values[counts.argmax()]
            fill_pos = np.less(filter_diff, common_value)
            filter_diff[fill_pos] = common_value
            enorm = 2.0 / filter_diff
        mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        warnings.warn(
            "At least one mel filter has all zero values. "
            f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
            f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
        )

    return mel_filters

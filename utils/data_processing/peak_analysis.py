import numpy as np

from scipy.signal import find_peaks
from scipy.stats import linregress


def detect_peaks_and_slopes(data, attack_type, window_days=[3, 7, 14], prominence_percentile=75):
    """
    Detect peaks in attack frequency and calculate slopes in the days/weeks before each peak.
    
    Parameters:
    - data: DataFrame with Date and Count columns for a specific attack type
    - attack_type: Name of the attack type
    - window_days: List of window sizes (in days) to analyze before peaks
    - prominence_percentile: Percentile for peak prominence threshold
    
    Returns:
    - Dictionary with peak information and slope analysis
    """
    counts = data['Count'].values
    dates = data['Date'].values
    
    # Detect peaks - prominence ensures we only get significant peaks
    prominence_threshold = np.percentile(counts, prominence_percentile) - np.median(counts)
    peaks, properties = find_peaks(counts, prominence=max(1, prominence_threshold))
    
    results = {
        'attack_type': attack_type,
        'num_peaks': len(peaks),
        'peak_dates': dates[peaks],
        'peak_values': counts[peaks],
        'slopes_analysis': {}
    }
    
    # For each window size, calculate slopes before peaks
    for window in window_days:
        slopes = []
        slopes_normalized = []
        
        for peak_idx in peaks:
            # Make sure we have enough data before the peak
            if peak_idx >= window:
                # Get data from the window before the peak
                window_data = counts[peak_idx - window:peak_idx]
                x = np.arange(len(window_data))
                
                # Calculate linear regression slope
                if len(window_data) > 1 and np.std(window_data) > 0:
                    slope, intercept, r_value, p_value, std_err = linregress(x, window_data)
                    slopes.append(slope)
                    
                    # Normalize slope by the mean to get percentage change per day
                    mean_val = np.mean(window_data) if np.mean(window_data) > 0 else 1
                    normalized_slope = (slope / mean_val) * 100
                    slopes_normalized.append(normalized_slope)
        
        results['slopes_analysis'][f'{window}d'] = {
            'slopes': slopes,
            'slopes_normalized': slopes_normalized,
            'mean_slope': np.mean(slopes) if slopes else 0,
            'mean_slope_normalized': np.mean(slopes_normalized) if slopes_normalized else 0,
            'positive_slopes': sum(1 for s in slopes if s > 0),
            'negative_slopes': sum(1 for s in slopes if s < 0),
            'positive_ratio': (sum(1 for s in slopes if s > 0) / len(slopes) * 100) if slopes else 0
        }
    
    return results
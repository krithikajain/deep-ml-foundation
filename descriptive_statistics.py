import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # 1. convert to numpy array
    arr = np.array(data, dtype=float)

    #2. calc central tendency
    mean = np.mean(arr)
    median = np.median(arr)

    #calc mode since no direct function
    vals, counts = np.unique(arr, return_counts=True)
    #find index of maximum counts
    mode_index = np.argmax(counts)
    mode_value = vals[mode_index]

    #3. calc the spread
    # ddof=0 means "Population" variance (divide by N), which is requested
    variance = np.var(arr, ddof=0)
    std_dev = np.std(arr, ddof=0)

    #4. percentiles
    p25 = np.percentile(arr, 25)
    p50 = np.percentile(arr, 50)
    p75 = np.percentile(arr, 75)

    #5. interquartile range (IQR)
    iqr = p75 - p25

    return {
        "mean": round(mean,4),
        "median": round(median,4),
        "mode": int(mode_value),
        "variance": round(variance,4),
        "standard_deviation": round(std_dev,4),
        "25th_percentile": round(p25,4),
        "50th_percentile": round(p50,4),
        "75th_percentile": round(p75,4),
        "interquartile_range": round(iqr,4)
    }

# Manual Check
if __name__ == "__main__":
    test_data = [1, 2, 2, 3, 4, 4, 4, 5]
    print(descriptive_statistics(test_data))
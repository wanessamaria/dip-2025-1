import numpy as np

def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    """
        Calculates the histogram intersection similarity between two grayscale images.

    This function measures how similar the pixel intensity distributions of 
    two grayscale images are by evaluating the overlap between their 
    normalized histograms with 256 bins.

    The histogram intersection is computed as the sum of the smallest values 
    for each matching bin in the two histograms. The score ranges from 0.0 
    (no common values) to 1.0 (identical histograms).

    Parameters:
        img1 (np.ndarray): First grayscale image as a 2D NumPy array.
        img2 (np.ndarray): Second grayscale image as a 2D NumPy array.

    Returns:
        float: Value in the range [0.0, 1.0] representing the histogram 
               intersection similarity.

    Raises:
        ValueError: If any of the inputs is not a 2D array (non-grayscale image).
    """    
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    ### START CODE HERE ###
    hist1, _ = np.histogram(img1, bins=256, range=(0, 255), density=True)
    hist2, _ = np.histogram(img2, bins=256, range=(0, 255), density=True)
    intersection = np.sum(np.minimum(hist1, hist2))
    ### END CODE HERE ###

    return float(intersection)

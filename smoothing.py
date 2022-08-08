import numpy as np


def savitzky_golay(t, x, order, window_size):
    # generalized implementation of the Savitzky-Golay filter for data smoothing
    # compared to Scipy's implementation, this function is slower but can take data
    # with non-uniform temporal spacing
    half_window = (window_size - 1) // 2

    first_x_vals = x[0] - np.abs(x[1:half_window + 1][::-1] - x[0])
    last_x_vals = x[-1] + np.abs(x[-half_window - 1:-1][::-1] - x[-1])

    t_spacing = (t[-1] - t[0]) / len(t)
    first_t_vals = np.array([-t_spacing * i + t[0] for i in range(1, half_window + 1)])
    last_t_vals = np.array([t_spacing * i + t[-1] for i in range(1, half_window + 1)])

    x_padded = np.concatenate((first_x_vals, x, last_x_vals))
    t_padded = np.concatenate((first_t_vals, t, last_t_vals))

    smoothed = []

    for i in range(len(t)):
        s = t_padded[i: i + window_size]
        y = x_padded[i: i + window_size]

        center = s[half_window]
        time_deltas = np.array([j - center for j in s])

        a = np.mat([[d ** m for m in range(order + 1)] for d in time_deltas])
        coeffs = np.linalg.pinv(a).A @ np.transpose(y)

        smoothed.append(coeffs[0])

    return np.array(smoothed)

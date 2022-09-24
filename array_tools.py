# this function will return the n indices with the closest values to a in an array arr

def nearest_items(a, arr, n):
    diff_arr = arr - a
    diff_abs = np.abs(diff_arr)
    min_abs = sorted(diff_abs)[:n]
    
    nearest = np.array([])
    for m in min_abs:
        if a - m in arr:
            nearest = np.append(nearest, a - m)
        if a + m in arr:
            nearest = np.append(nearest, a + m)

    # previous values are slightly prioritized over future values in cases of high symmetry
    return nearest[:n] 

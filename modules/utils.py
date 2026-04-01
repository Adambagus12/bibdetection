def add_padding(x1, y1, x2, y2, frame_shape, pad=15):
    h, w, _ = frame_shape

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return x1, y1, x2, y2
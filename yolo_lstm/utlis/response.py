def build_response(label="-", confidence=0.0, mode_name="LSTM", bbox=None):
    return {
        "label": label,
        "confidence": confidence,
        "mode": mode_name,
        "bbox": bbox,
    }

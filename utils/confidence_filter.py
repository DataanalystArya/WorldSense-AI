def filter_by_confidence(results, threshold=0.12):
    return [
        r for r in results
        if r["confidence"] >= threshold
    ]

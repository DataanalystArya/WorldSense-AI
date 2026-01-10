def confusion_aware_scene(scene_result):
    """
    Adds human-like confidence awareness on top of existing scene output.
    Does NOT change original pipeline.
    """

    if "top_candidates" not in scene_result:
        return scene_result

    top = scene_result["top_candidates"][0]
    second = scene_result["top_candidates"][1]

    best_score = top["score"]
    second_score = second["score"]

    diff_ratio = (best_score - second_score) / max(best_score, 1e-6)

    if diff_ratio >= 0.25:
        confidence_level = "HIGH"
    elif diff_ratio >= 0.10:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return {
        "scene": scene_result["scene"],
        "confidence_level": confidence_level,
        "confidence_score": round(diff_ratio, 3),
        "top_candidates": scene_result["top_candidates"]
    }

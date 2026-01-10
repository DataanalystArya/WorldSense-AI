def generate_why_explanation(scene_name, confidence_level):
    """
    Generate human-readable WHY explanation
    This is a reasoning layer, not model truth
    """

    base_reasons = [
        "Overall visual patterns match the predicted scene",
        "Color, lighting, and spatial layout are consistent",
        "Detected objects support this environment type"
    ]

    scene_specific_hints = {
        "traffic road": [
            "Road-like structure visible",
            "Presence of multiple vehicles",
            "Outdoor urban environment detected"
        ],
        "outdoor street": [
            "Open outdoor space detected",
            "Street-like layout visible"
        ],
        "night time urban scene": [
            "Low-light conditions observed",
            "Artificial lighting patterns detected"
        ],
        "crowded public place": [
            "Multiple people detected",
            "Dense activity in the scene"
        ]
    }

    reasons = []

    if scene_name in scene_specific_hints:
        reasons.extend(scene_specific_hints[scene_name][:2])

    reasons.extend(base_reasons)

    if confidence_level == "LOW":
        reasons.append("Confidence is low due to similarity with other scenes")

    return reasons[:4]

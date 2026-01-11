def generate_event_summary(scene_result):
    scene = scene_result.get("scene", "unknown scene")
    confidence = scene_result.get("confidence_level", "LOW")
    top_candidates = scene_result.get("top_candidates", [])

    # Smarter scene interpretation
    if scene in ["traffic road", "outdoor street"]:
        scene_text = "a road-based traffic environment, likely a highway or major roadway"
    elif scene == "crowded public place":
        scene_text = "a crowded public environment with significant human activity"
    else:
        scene_text = f"a {scene}"

    if confidence == "HIGH":
        confidence_text = "with high confidence"
    elif confidence == "MEDIUM":
        confidence_text = "with moderate confidence"
    else:
        confidence_text = "but confidence is low due to similarity with nearby scene categories"

    summary = (
        f"This video appears to show {scene_text}. "
        f"The system identifies this scene {confidence_text}. "
    )

    if top_candidates:
        alternatives = ", ".join(
            [c["scene"] for c in top_candidates[1:]]
        )
        summary += (
            f"Other visually similar scenes include {alternatives}."
        )

    return summary

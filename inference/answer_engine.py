def generate_answer(intent, objects, scene):
    objects = set(objects)

    if intent == "describe":
        return f"This image shows a {scene} with objects like {', '.join(objects)}."

    if intent == "crowd_check":
        if "crowd" in objects or "person" in objects:
            return "Yes, there appears to be a crowd in the scene."
        else:
            return "No significant crowd is visible."

    if intent == "risk_analysis":
        if scene in ["highway", "city street"] and "vehicle" in objects:
            return "This scene may involve traffic-related risk."
        return "No obvious danger detected."

    if intent == "scene":
        return f"The scene appears to be a {scene}."

    return "I am not confident enough to answer that question."

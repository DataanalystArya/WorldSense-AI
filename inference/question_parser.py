def parse_question(question):
    q = question.lower()

    if any(k in q for k in ["what", "kya", "describe"]):
        return "describe"

    if any(k in q for k in ["crowd", "bheed"]):
        return "crowd_check"

    if any(k in q for k in ["danger", "safe", "risk"]):
        return "risk_analysis"

    if any(k in q for k in ["where", "place"]):
        return "scene"

    return "unknown"

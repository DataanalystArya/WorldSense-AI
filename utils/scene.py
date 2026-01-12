def infer_scene(counts):
    persons = counts.get("person", 0)
    vehicles = sum(counts.get(v, 0) for v in ["car", "bus", "truck", "motorcycle"])

    if persons >= 6:
        return " Crowd Detected", "High Risk Public Area"
    elif vehicles > persons:
        return " Traffic Scene", "Urban / Highway Monitoring"
    else:
        return " General Surveillance", "Normal Activity"

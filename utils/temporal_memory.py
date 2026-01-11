class TemporalMemory:
    def __init__(self):
        self.timeline = []

    def update(self, scene_result):
        self.timeline.append({
            "scene": scene_result["scene"],
            "confidence": scene_result["confidence_level"]
        })

    def summary(self):
        scene_count = {}
        for item in self.timeline:
            scene = item["scene"]
            scene_count[scene] = scene_count.get(scene, 0) + 1

        most_common = max(scene_count, key=scene_count.get)

        return {
            "total_segments": len(self.timeline),
            "most_frequent_scene": most_common,
            "scene_timeline": self.timeline
        }

    def ask(self, question):
        q = question.lower()

        if "most" in q:
            return self.summary()["most_frequent_scene"]

        if "how many" in q:
            return len(self.timeline)

        if "timeline" in q:
            return self.timeline

        return "Question not understood yet"

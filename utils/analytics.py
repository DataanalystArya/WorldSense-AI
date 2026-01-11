import time
from collections import defaultdict

class Analytics:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def update_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time

        if elapsed >= 1.0:
            self.fps = round(self.frame_count / elapsed, 2)
            self.start_time = time.time()
            self.frame_count = 0

        return self.fps

    def count_objects(self, detections, class_names):
        counts = defaultdict(int)

        for box in detections:
            cls_id = int(box.cls)
            class_name = class_names[cls_id]
            counts[class_name] += 1

        return dict(counts)

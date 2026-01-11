import time
import os

class AlertSystem:
    def __init__(self, person_threshold=5, cooldown=3):
        self.person_threshold = person_threshold
        self.cooldown = cooldown
        self.last_alert_time = 0

    def can_alert(self):
        return time.time() - self.last_alert_time > self.cooldown

    def play_sound(self):
        os.system("afplay /System/Library/Sounds/Glass.aiff")

    def trigger_alert(self, message):
        if self.can_alert():
            print(f"🚨 ALERT: {message}")
            self.play_sound()
            self.last_alert_time = time.time()

    def check_alerts(self, counts):
        if counts.get("person", 0) > self.person_threshold:
            self.trigger_alert(
                f"High crowd detected ({counts['person']} persons)"
            )
            return True
        return False

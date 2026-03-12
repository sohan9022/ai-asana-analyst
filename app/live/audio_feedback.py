import pyttsx3
import threading
import queue
import time

class AudioFeedback:
    def __init__(self):
        self.queue    = queue.Queue()
        self.muted    = False
        self.cooldown = 4.0  # Slightly longer cooldown so the user has time to adjust
        self.last_spoken = {}

        # Single dedicated thread for audio — no conflicts
        self.thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.thread.start()

    def _audio_worker(self):
        """Single background thread that processes audio queue one at a time."""
        # Note for MacOS users: sometimes pyttsx3.init() needs to be inside the while loop
        engine = pyttsx3.init()
        engine.setProperty("rate", 165) # Slightly faster so it feels real-time
        engine.setProperty("volume", 1.0)

        while True:
            try:
                text = self.queue.get(timeout=1)
                if not self.muted:
                    engine.say(text)
                    engine.runAndWait()
                self.queue.task_done()
            except queue.Empty:
                continue

    def process_violations(self, violations):
        """Add violation alerts to queue, prioritizing one correction at a time."""
        if not violations:
            return
            
        now = time.time()
        
        # ── COACHING FIX ──
        # Only coach the user on the FIRST (most critical) mistake to avoid overwhelming them
        v = violations[0] 
        joint = v["joint"]
        last  = self.last_spoken.get(joint, 0)

        if now - last >= self.cooldown:
            # Drop the mistake part to make it sound more encouraging, just give the correction
            message = v['correction'] 
            
            if self.queue.qsize() < 2:
                self.queue.put(message)
            self.last_spoken[joint] = now

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted
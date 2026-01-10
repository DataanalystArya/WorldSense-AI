import speech_recognition as sr

def listen_question():
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 Speak now...")
            audio = r.listen(source)
        return r.recognize_google(audio)
    except Exception:
        # fallback to text
        return input("❓ Ask a question: ").strip()

import pyttsx3

engine = pyttsx3.init()

def speak(text):
    print(" AI:", text)
    engine.say(text)
    engine.runAndWait()

import os
from inference.detect_image_zeroshot import analyze_image
from inference.question_parser import parse_question
from inference.answer_engine import generate_answer
from voice.voice_input import listen_question
from voice.voice_output import speak

image_path = input("📷 Enter image path: ").strip()

if not os.path.exists(image_path):
    speak("Image file not found.")
    exit()

objects, scene = analyze_image(image_path)

print("🧠 Image analyzed.")
print("Objects:", objects)
print("Scene:", scene)

# 🎤 ONLY VOICE — NO TEXT FALLBACK
question = listen_question()

if question is None:
    speak("I could not hear you. Please speak again.")
    exit()

print("❓ Question:", question)

intent = parse_question(question)
answer = generate_answer(intent, objects, scene)

speak(answer)

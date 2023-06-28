import pyttsx3


def say(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty("voices")
    engine.setProperty("voices", voices[1].id)
    engine.setProperty("rate", 170)
    print(f"A.I : {Text}")
    engine.say(text=Text)
    engine.runAndWait()
    print("   ")

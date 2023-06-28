import speech_recognition as sr


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source, 0, 4)
    try:
        print("Recognizing....")
        query = r.recognize_google(audio, language="en-in")
        print(f"You said: {query}")

    except:
        print("")

    query = str(query)
    return query.lower()

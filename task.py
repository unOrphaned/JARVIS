import datetime
from speak import say


def time():
    time = datetime.datetime.now().strftime("%H:%M:%S")
    say(time)


def date():
    date = datetime.date.today()
    say(date)


def day():
    day = datetime.datetime.now().strftime("%A")
    say(day)


def noninputexcute(query):
    query = str(query)

    if "time" in query:
        time()
    elif "date" in query:
        date()
    elif "day" in query:
        day()


def inputexecution(tag, query):
    if "wikipedia" in tag:
        name = str(query).replace("who is", "").replace("tell me about", "").replace("wikipedia", "").replace("gather information", "")
        import wikipedia
        result = wikipedia.summary(name)
        say(result)
    elif "google" in tag:
        query = str(query).replace("google", "")
        query = query.replace("search", "")
        import pywhatkit
        pywhatkit.search(query)
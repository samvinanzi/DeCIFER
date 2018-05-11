"""

Thread that will listen to the default microphone and return detected vocal commands for the robot.

"""

import speech_recognition as sr
from threading import Thread
from queue import Queue
from asyncio import QueueFull


class Listener(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.stop_flag = False
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()   # Default device
        if isinstance(queue, Queue):
            self.queue = queue
        else:
            print("[ERROR] Invalid Queue reference in Listener initialization.")

    def run(self):
        while not self.stop_flag:
            # adjust the recognizer sensitivity to ambient noise and record audio from the microphone
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
            try:
                response = self.recognizer.recognize_google(audio)
                status = "Ok"
            except sr.RequestError:
                # API was unreachable or unresponsive
                response = None
                status = "API unavailable"
            except sr.UnknownValueError:
                # speech was unintelligible
                response = None
                status = "Unable to recognize speech"
            #return response, status
            try:
                self.queue.put([response, status])
            except QueueFull:
                print("[ERROR] Queue item is full and cannot accept further insertions.")
        print("[DEBUG] Shutting down Listener thread.")

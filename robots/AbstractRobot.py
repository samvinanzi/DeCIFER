"""

Abstract robot class that serves as a superclass for the specific platform implementations

"""


import socket
from abc import ABC, abstractmethod
from threading import Lock, Event
import time
import pyttsx3 as tts
import speech_recognition as sr


class AbstractRobot(ABC):
    def __init__(self):
        self.remote_listener_ip = '127.0.0.1'    # ToDo set ! ! ! !
        self.remote_listener_port = 50106
        # Initializations
        self.vocal_queue = []
        self.accepted_commands = ["START", "STOP", "YES", "NO"]
        self.tts = tts.init()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()  # Default device
        # Synchronization
        self.lock = Lock()      # to synchronize access to vocal_queue
        self.event = Event()    # to signal the presence of new data in vocal_queue
        # Configurations
        self.tts.setProperty('rate', 140)                   # Text to Speech
        self.tts.setProperty('voice', 'english')
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)  # we only need to calibrate once before we start listening
        # For development purposes only
        # Ordered list of vocal responses to the robot's questions
        self.command_list = []
        self.command_list.reverse()
        self.AUTORESPONDER_ENABLED = False
        self.coordinates = {}

    # Text to Speech
    def say(self, phrase):
        self.tts.say(phrase)
        print("[DEBUG] Robot says: " + phrase)
        self.tts.runAndWait()

    @abstractmethod
    # Retrieve platform-dependent image containers
    def get_image_containers(self):
        pass

    @abstractmethod
    # Takes an object
    def action_take(self, coordinates):
        pass

    @abstractmethod
    # Points to an object
    def action_point(self, coordinates):
        pass

    @abstractmethod
    # Gives an object
    def action_give(self):
        pass

    @abstractmethod
    # Requests an object
    def action_expect(self):
        pass

    @abstractmethod
    # Returns in home position
    def action_home(self):
        pass

    @abstractmethod
    # Looks at one specific direction
    def action_look(self, coordinates):
        pass

    @abstractmethod
    # Drops an object
    def action_drop(self, coordinates):
        pass

    @abstractmethod
    # Looks for a skeleton in a given image frame. Can raise NoHumansFoundException
    def look_for_skeleton(self, image_containers, i):
        pass

    @abstractmethod
    # Searches for an object
    def search_for_object(self):
        pass

    @abstractmethod
    # Evaluates the construction
    def evaluate_construction(self):
        pass

    @abstractmethod
    # Determines the color of the object, provided the bounding box that encloses it
    def get_color(self):
        pass

    @abstractmethod
    # Closes all the open ports
    def cleanup(self):
        pass

    # --- SPEECH RECOGNITION METHODS --- #

    # Analyses the vocal string in search of known commands or for a specific command
    def recognize_commands(self, response, listenFor=None):
        if response is None:
            return None
        else:
            response = response.upper()
            if response in self.accepted_commands and (listenFor is None or response == listenFor):
                return response
            else:
                return None

    # This function is called by the background listener thread, if running, when a new audio signal is detected
    def speech_recognition_callback(self, recognizer, audio):
        print("[DEBUG] Detected audio. Recognizing...")
        self.say("Ok")
        try:
            response = self.recognizer.recognize_google(audio, show_all=True)
            with self.lock:  # In case of exception, this lock won't be opened
                self.vocal_queue.append(response)
            self.event.set()
        except sr.UnknownValueError:
            print("[DEBUG] Google Speech Recognition could not understand audio")
            self.say("Sorry, I didn't understand. Can you please repeat?")
        except sr.RequestError as e:
            print("[DEBUG] Could not request results from Google Speech Recognition service; {0}".format(e))
            self.say("Sorry, I didn't understand. Can you please repeat?")

    # Listens for valid vocal input (commands are not processed at this stage, but None responses are discarded as
    # they are founded in the queue)
    def wait_and_listen(self):
        # Starts the background recording
        stop_listening = self.recognizer.listen_in_background(self.microphone, self.speech_recognition_callback)
        print("[DEBUG] Listening in background")
        # non-busy-waiting for the listener to signal the production of new data
        self.event.wait()
        with self.lock:
            response = self.vocal_queue.pop()  # Secure consumption of the data
        self.event.clear()
        stop_listening(wait_for_stop=True)  # Makes sure that the background thread has stopped before continuing
        print("[DEBUG] Listening stopped")
        return response

    # Only for debugging purposes
    def wait_and_listen_dummy(self):
        if self.AUTORESPONDER_ENABLED:
            time.sleep(2)
            response = self.command_list.pop()
            print("[DEBUG] Autoresponse: " + response)
        else:
            response = input("Digit the word you would pronounce: ")
        return response

    # Remote microphone service (to use on iCub with no microphones onboard)
    def wait_and_listen_remote(self, ip=None):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if ip is not None:
                client_socket.connect((ip, self.remote_listener_port))
                print("Connected to " + ip + ":" + str(self.remote_listener_port))
            else:
                client_socket.connect((self.remote_listener_ip, self.remote_listener_port))
                print("Connected to " + self.remote_listener_ip + ":" + str(self.remote_listener_port))
            client_socket.send('listen'.encode('utf-8'))
            print("[DEBUG] Sent request, waiting for response...")
            response = client_socket.recv(1024).decode('utf-8')
        except socket.error as e:
            print("[ERROR] Connection error while connecting to " + self.remote_listener_ip)
            response = None
        client_socket.close()
        print("[DEBUG] Received: " + response)
        return response

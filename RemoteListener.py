"""

Remote listener service that runs on the iCub-Desktop machine and waits for a trigger to capture microphone audio.
Start the service on that machine before running the DeCIFER experiments.

"""

import speech_recognition as sr
from threading import Lock, Event
import socket


class RemoteListener:
    def __init__(self):
        # Networking parameters
        self.HOST = '127.0.0.1'     # todo change with icub-desktop static ip
        self.PORT = 50106
        self.SOCKSIZE = 1024

        # Task variables
        self.vocal_queue = []
        self.lock = Lock()          # to synchronize access to vocal_queue
        self.event = Event()        # to signal the presence of new data in vocal_queue
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()  # Default device
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)  # we only need to calibrate once before we start listening
        print("****\nRemoteListener is active. Ctrl-C to shutdown.")

    # This function is called by the background listener thread, if running, when a new audio signal is detected
    def speech_recognition_callback(self, recognizer, audio):
        print("[DEBUG] Detected remote audio. Recognizing...")
        try:
            response = self.recognizer.recognize_google(audio)
            with self.lock:  # In case of exception, this lock won't be opened
                self.vocal_queue.append(response)
            self.event.set()
        except sr.UnknownValueError:
            print("[DEBUG] Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("[DEBUG] Could not request results from Google Speech Recognition service; {0}".format(e))

    # Listens for valid vocal input (commands are not processed at this stage, but None responses are discarded as
    # they are founded in the queue)
    def wait_and_listen(self):
        # Starts the background recording
        stop_listening = self.recognizer.listen_in_background(self.microphone, self.speech_recognition_callback)
        print("[DEBUG] Listening remotely in background")
        # non-busy-waiting for the listener to signal the production of new data
        self.event.wait()
        with self.lock:
            response = self.vocal_queue.pop()  # Secure consumption of the data
        self.event.clear()
        stop_listening(wait_for_stop=True)  # Makes sure that the background thread has stopped before continuing
        print("[DEBUG] Remote listening stopped")
        return response

    # Waits for a network request to start processing audio, fetches data and returns them to the client
    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Should reuse the same address if aborted
            server_socket.bind((self.HOST, self.PORT))
            print("Server started on port: %s" % self.PORT)
            server_socket.listen(1)
            print("Now waiting for connections...\n")
            while True:
                client_socket, client_address = server_socket.accept()
                print('New connection from %s:%d' % (client_address[0], client_address[1]))
                data = client_socket.recv(self.SOCKSIZE).decode('utf-8')
                if not data:
                    print("No data received!")
                elif data == 'listen':
                    print("Listening...")
                    response = self.wait_and_listen()
                    print("Sending response \"" + response + "\" to client " + client_address[0] + ":" + str(client_address[1]))
                    client_socket.send(response.encode('utf-8'))
                elif data == 'kill':
                    print("Killing...")
                    raise KeyboardInterrupt
                else:
                    print("Received invalid command: " + str(data))
        except KeyboardInterrupt:
            print("\nRemoteListener server closing.")
            server_socket.close()


# -- Instantiates an object

RemoteListener().start()

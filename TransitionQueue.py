"""

This class models a transition queue made up by two synchronized elements: a Queue and an Event.

"""

from queue import Queue
from asyncio import QueueEmpty, QueueFull
from threading import Event


class TransitionQueue:
    def __init__(self):
        self.queue = Queue()    # contains the cluster transitions
        self.event = Event()    # signals the presence of new data in vocal_queue

    # Inserts an item in the queue and signals the event
    def put(self, item):
        try:
            self.queue.put(item)
        except QueueFull:
            print("[ERROR] Queue full, unable to fullfil latest insertion request")
        self.event.set()

    # Waits for an event to occour, then tries to retrieve the item and signals the operation as completed
    def get(self):
        self.event.wait()
        if not self.queue.empty():
            try:
                item = self.queue.get()
                self.queue.task_done()
                self.event.clear()
                return item
            except QueueEmpty:
                print("[ERROR] Invalid access to empty transition queue")

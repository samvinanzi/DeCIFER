"""

This class represents an intention, that as stated in literature is "a goal plus an action plan".

"""


class Intention:
    def __init__(self):
        self.actions = []   # Ordered cluster transitions
        self.goal = None

    def has_goal(self):
        return True if self.goal is not None else False

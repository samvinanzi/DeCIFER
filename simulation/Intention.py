"""

This class represents an intention, that as stated in literature is "a goal plus an action plan".

"""


class Intention:
    def __init__(self):
        self.actions = []   # Ordered cluster transitions
        self.goal = None

    # Has this intention have a goal label?
    def has_goal(self):
        return True if self.goal is not None else False

    # Produces a dictionary
    def as_dict(self):
        return {
            'data': self.actions,
            'label': self.goal
        }

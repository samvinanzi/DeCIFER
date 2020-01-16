"""

Transition buffer to filter Near-to-Far Transitions (NFT).

"""

import re


class Buffer:
    def __init__(self, debug=False):
        self.buffer = []
        self.output = []
        self.debug = debug

    def detect_nft(self):
        # Transforms the list in a string sequence
        string_sequence = ""
        for number in self.buffer:
            string_sequence += str(number)
        # Checks for a NFT for left or right side
        p = re.compile('(4+3)|(6+5)')
        m = p.match(string_sequence)
        if m:
            return True
        else:
            return False
    
    def insert(self, token):
        # If a non-0 symbol is encountered, add it to the buffer
        if token != 0:
            self.buffer.append(token)
            if self.debug:
                print("[DEBUG] Adding " + str(token) + " to the buffer")
        # Otherwise, evaluate the symbols contained in the buffer
        else:
            if self.debug:
                print("[DEBUG] Termination symbol encountered")
            # Empty buffer, do nothing
            if len(self.buffer) == 0:
                pass
            # Buffer contains a single element: return it
            elif len(self.buffer) == 1:
                self.output = self.buffer
            # Buffer contains a sequence of symbols. Evaluate them
            elif len(self.buffer) > 1:
                # Do the symbols represent a NFT?
                result = self.detect_nft()
                # Yes, retrieve the "far" symbol (the one with the smaller id)
                if result:
                    self.output = [min(self.buffer)]
                # No, retrieve all the unique symbols
                else:
                    [self.output.append(item) for item in self.buffer if item not in self.output]
                buffer = []     # Empty the buffer
            if len(self.output) > 0:
                # If the output contains something, return it
                if self.debug:
                    print("[DEBUG] Buffer is returning values: " + str(self.output))
                return self.output

import numpy as np
class Matrix:
    data = None

    def __init__(self, data):
        self.data = np.array(data)

    def __str__(self):
        ret = "[Matrix]\n"
        for row in self.data:
            ret += str([("%5d" % col) for col in row])
            ret += "\n"
        return ret

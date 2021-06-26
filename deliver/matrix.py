class Matrix:
    data = []

    def __init__(self, data):
        self.data = data

    def __str__(self):
        ret = "[Matrix]\n"
        for row in self.data:
            ret += str([("%5d" % col) for col in row])
            ret += "\n"
        return ret

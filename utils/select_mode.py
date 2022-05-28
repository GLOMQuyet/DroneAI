class Select:
    def __init__(self,key,mode):
        self.key = key
        self.mode = mode

    def select_mode(self):
        self.number = -1
        if 48 <= self.key <= 57:  # 0 ~ 9
            self.number = self.key - 48
        if self.key == 110:  # n
            self.mode = 0
        if self.key == 107:  # k
            self.mode = 1
        if self.key == 104:  # h
            self.mode = 2
        return self.number, self.mode
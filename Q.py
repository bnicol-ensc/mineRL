reward = 0


class Q:
    dico = {}
    learningRate = 0.8
    discountFactor = 0.5


    def Qtest(self, key):
        if(key in self.dico):
            return self.dico.get(key)
        else:
            self.dico[key] = 0
            return 0

    def Qlearning(self, key):
        key2 = 1
        return self.Qtest(key) + self.learningRate * (reward + self.discountFactor * self.Qtest(key2)-self.Qtest(key))
class Algorithms:
    def __init__(self, algorithms={}, schedulers={}):
        self.algorithms = algorithms
        self.schedulers = schedulers

    def registerAlgorithm(self, key, algorithm):
       self.algorithms[key] = algorithm

    def getAlgorithm(self, key):
        return self.algorithms[key]
    
    def registerScheduler(self, key, scheduler):
       self.schedulers[key] = scheduler

    def getScheduler(self, key):
        return self.schedulers[key]
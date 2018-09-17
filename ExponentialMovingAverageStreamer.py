import numpy as np

class ExponentialMovingAverageStreamer:
    def __init__(self, period):
        self.period = period
        self.beta = 2 / (period + 1)
        self.lastValue = 0

    def setup(self, initialData):
        if len(initialData) < self.period:
            raise(ValueError("Not enough initialization data."))

        simpleAverage = np.mean(initialData[:self.period])

        self.lastValue = simpleAverage

        for data in initialData[self.period:]:
            self.onData(data)

    def onData(self, data):
        self.lastValue = self.beta * data + (1 - self.beta) * self.lastValue

        return self.lastValue

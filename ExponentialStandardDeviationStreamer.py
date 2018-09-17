import numpy as np

from ExponentialMovingAverageStreamer import ExponentialMovingAverageStreamer

class ExponentialStandardDeviationStreamer:
    def __init__(self, period):
        self.period = period
        self.beta = 2 / (period + 1)
        self.emaStreamer = ExponentialMovingAverageStreamer(period)
        self.lastValue = 0

    def setup(self, initialData):
        if len(initialData) < self.period:
            raise(ValueError("Not enough initialization data."))

        simpleStd = np.std(initialData[:self.period])

        self.lastValue = simpleStd

        self.emaStreamer.setup(initialData)

        for data in initialData[self.period:]:
            self.onData(data)
            self.emaStreamer.onData(data)

    def onData(self, data):

        lastEMV= pow(self.lastValue, 2)
        EMV = (1 - self.beta) * (lastEMV + self.beta * pow(data - self.emaStreamer.lastValue, 2))
        self.lastValue = pow(EMV, 0.5)

        return self.lastValue

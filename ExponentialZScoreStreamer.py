import numpy as np

from ExponentialStandardDeviationStreamer import ExponentialStandardDeviationStreamer

class ExponentialZScoreStreamer:
    def __init__(self, period):
        self.estdStreamer = ExponentialStandardDeviationStreamer(period)
        self.lastValue = 0

    def setup(self, initialData):
        self.estdStreamer.setup(initialData)

        self.lastValue = (initialData[-1] - self.estdStreamer.emaStreamer.lastValue) \
            / self.estdStreamer.lastValue


    def onData(self, data):
        self.estdStreamer.onData(data)

        self.lastValue = (data - self.estdStreamer.emaStreamer.lastValue) \
            / self.estdStreamer.lastValue

        return self.lastValue

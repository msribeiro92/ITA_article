class MovingAverageStreamer:
    def __init__(self, period):
        self.period = period
        self.buffer = []
        self.lastValue = 0

    def setup(self, initialData):
        if len(initialData) < self.period:
            raise(ValueError("Not enough initialization data."))

        for data in initialData:
            self.buffer.append(data)
            if len(self.buffer) > self.period:
                self.buffer.pop(0)

        movingAverage = sum(self.buffer)
        movingAverage /= self.period

        self.lastValue = movingAverage

    def onData(self, data):
        self.buffer.append(data)
        self.buffer.pop(0)

        movingAverage = sum(self.buffer)
        movingAverage /= self.period

        self.lastValue = movingAverage

        return movingAverage

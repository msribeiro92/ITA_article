from MovingAverageStreamer import MovingAverageStreamer

class MovingAverageCrossing:

    def __init__(self, shortPeriod, longPeriod):
        self.shortAverage = MovingAverageStreamer(shortPeriod)
        self.shortPeriod = shortPeriod
        self.longAverage = MovingAverageStreamer(longPeriod)
        self.longPeriod = longPeriod
        self.trend = False  #False: downtrend; True: uptrend
        self.lastValue = (False, True)

    def getInitializationSize(self):
        return self.longPeriod

    def setup(self, initialData):
        if len(initialData) < self.getInitializationSize():
            raise(ValueError("Not enough initialization data."))

        self.shortAverage.setup(initialData)
        self.longAverage.setup(initialData)

        shortAverage = self.shortAverage.lastValue
        longAverage = self.longAverage.lastValue

        if shortAverage > longAverage:
            trend = True
            reversal = trend != self.trend
        elif shortAverage < longAverage:
            trend = False
            reversal = trend != self.trend
        else:
            trend = self.trend
            reversal = False

        self.trend = trend
        self.lastValue = (trend, reversal)

    def onData(self, data):
        shortAverage = self.shortAverage.onData(data)
        longAverage = self.longAverage.onData(data)

        if shortAverage > longAverage:
            trend = True
            reversal = trend != self.trend
        elif shortAverage < longAverage:
            trend = False
            reversal = trend != self.trend
        else:
            trend = self.trend
            reversal = False

        self.trend = trend
        self.lastValue = (trend, reversal)

        return (trend, reversal)

import pandas as pd
import numpy as np

from ExponentialZScoreStreamer import ExponentialZScoreStreamer

class DiscretizedEnvironment:

    def __init__(self, stock="AAPL", initializationPeriod=10):
        dataFrame = pd.read_csv(
            'spx_intraday.csv', index_col=0, header=[0, 1]
        ).sort_index(axis=1)

        self.initializationPeriod = initializationPeriod
        self.priceData = dataFrame[stock]["close"]
        self.volumeData = dataFrame[stock]["volume"]
        self.price = 0 # mean price of the position as per historical trades
        self.position = 0 # Number of lots

        self.action_space = np.array([-3, -2, -1, 0, 1, 2, 3])

        self.zStreamerPrice = ExponentialZScoreStreamer(initializationPeriod)
        self.zStreamerPrice.setup(self.priceData[:initializationPeriod])
        self.zStreamerVolume = ExponentialZScoreStreamer(initializationPeriod)
        self.zStreamerVolume.setup(self.volumeData[:initializationPeriod])
        self.index = initializationPeriod - 1

    def getCurrentState(self):
        discretizedZPrice = max(min(int(round(self.zStreamerPrice.lastValue)), 3), -3)
        discretizedZVolume = max(min(int(round(self.zStreamerVolume.lastValue)), 3), -3)

        # map to a 0-48 labeled state
        return (discretizedZPrice + 3) * 7 + (discretizedZVolume + 3)

    def reset(self):
        self.value = 0 # value of the current position
        self.position = 0 # Number of lots

        self.zStreamerPrice = ExponentialZScoreStreamer(self.initializationPeriod)
        self.zStreamerPrice.setup(self.priceData[:self.initializationPeriod])
        self.zStreamerVolume = ExponentialZScoreStreamer(self.initializationPeriod)
        self.zStreamerVolume.setup(self.volumeData[:self.initializationPeriod])
        self.index = self.initializationPeriod - 1

        return self.getCurrentState()

    def step(self, action):
        self.index += 1
        lastPos = self.position
        lastPrice = self.price
        lastValue = lastPos * lastPrice
        self.position += action # action = delta(pos)
        cash = action * self.priceData[self.index]
        if self.position > 0:
            self.price = (lastValue + cash) / self.position
        else:
            self.price = 0
        reward = (self.position * self.price - lastValue) - cash

        self.zStreamerPrice.onData(self.priceData[self.index])
        self.zStreamerVolume.onData(self.volumeData[self.index])
        state = self.getCurrentState()

        return(state, reward)

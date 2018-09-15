import pandas as pd
import numpy as np

class Environment:

    def __init__(self, stock="AAPL", memory=10):
        dataFrame = pd.read_csv(
            'spx_intraday.csv', index_col=0, header=[0, 1]
        ).sort_index(axis=1)

        self.data = dataFrame[stock]["close"]
        self.index = memory;
        self.price = 0;
        self.memory = memory

        self.actionDict = {0: "sell", 1: "buy"}
        self.action_space = np.array([0, 1])

        self.position = "unpositioned";
        self.positionDict = {"unpositioned": 0, "long": 1, "short": 2}

        if self.data[memory-1] >= self.data[0]:
            self.trend = "up"
        else:
            self.trend = "down"
        self.trendDict = {"down": 0, "up": 1}

        self.stateDict = {}
        state = 0;
        for trend in self.trendDict:
            for pos in self.positionDict:
                self.stateDict[trend + "," + pos] = state
                state += 1

    def reset(self):
        self.index = self.memory;
        self.price = 0;

        self.position = "unpositioned";

        if self.data[self.memory-1] >= self.data[0]:
            self.trend = "up"
        else:
            self.trend = "down"
        self.trendDict = {"down": 0, "up": 1}

        state = self.stateDict[self.trend + "," + self.position]
        return state

    def step(self, action):

        self.index += 1

        if self.data[self.index] >= self.data[self.index - self.memory ]:
            self.trend = "up"
        else:
            self.trend = "down"

        reward = 0;
        if self.actionDict[action] == "buy":
            if self.position == "unpositioned":
                self.position = "long"
                self.price = self.data[self.index]
            elif self.position == "short":
                self.position = "unpositioned"
                reward = self.price + self.data[self.index]
                self.price = self.data[self.index]
        if self.actionDict[action] == "sell":
            if self.position == "unpositioned":
                self.position = "short"
                self.price = -self.data[self.index]
            elif self.position == "long":
                self.position = "unpositioned"
                reward = self.price - self.data[self.index]
                self.price = self.data[self.index]

        state = self.stateDict[self.trend + "," + self.position]
        return(state, reward)

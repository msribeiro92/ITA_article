import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ReinforcementIndicator import ReinforcementIndicator

class Test:

    def __init__(self):
        self.dataFrame =  pd.read_csv('spx_intraday.csv', index_col=0, header=[0, 1]).sort_index(axis=1)

    def testSampleData(self, stock):
        self.dataFrame[stock]["close"].plot()
        plt.show()


#test = Test()
#test.testSampleData("AAPL")

ind = ReinforcementIndicator()
ind.train()

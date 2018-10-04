import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from ReinforcementIndicator import ReinforcementIndicator
from TableReinforcementIndicator import TableReinforcementIndicator

class Test:

    def __init__(self):
        self.dataFrame =  pd.read_csv('spx_intraday.csv', index_col=0, header=[0, 1]).sort_index(axis=1)

    def testSampleData(self, stock):
        self.dataFrame[stock]["close"][2000:4000].plot()
        print len(self.dataFrame[stock]["close"])
        plt.show()
        #print max(self.dataFrame[stock]["volume"])
        #print min(self.dataFrame[stock]["volume"])
        #print np.mean(self.dataFrame[stock]["volume"])



#print sys.argv


print sys.argv
ind = TableReinforcementIndicator(sys.argv[1])
ind.testSelected()
#test = Test()
#test.testSampleData(sys.argv[1])

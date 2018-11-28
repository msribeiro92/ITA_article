import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime

from ReinforcementIndicator import ReinforcementIndicator
from TableReinforcementIndicator import TableReinforcementIndicator

class Test:

    def __init__(self):
        self.dataFrame =  pd.read_csv('spx_intraday.csv', index_col=0, header=[0, 1]).sort_index(axis=1)

    def testSampleData(self, stock):
        self.dataFrame[stock]["close"][2000:4000].plot()
        print len(self.dataFrame[stock]["close"])
        #plt.show()
        fig_name = 'testfig.png'
        plt.savefig(fig_name)

        #print max(self.dataFrame[stock]["volume"])
        #print min(self.dataFrame[stock]["volume"])
        #print np.mean(self.dataFrame[stock]["volume"])



#print sys.argv


#print sys.argv
print datetime.datetime.now()
ind = TableReinforcementIndicator('AAPL')
#ind.testSelected()
#print datetime.datetime.now()
#ind = ReinforcementIndicator('AAPL')
#ind.testSelected()
#print datetime.datetime.now()

#test = Test()
#test.testSampleData(sys.argv[1])

#print datetime.datetime.now()
#ind = TableReinforcementIndicator('MSFT')
#ind.testSelected()
#print datetime.datetime.now()
#ind = ReinforcementIndicator('MSFT')
#ind.testSelected()
#print datetime.datetime.now()

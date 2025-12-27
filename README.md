# CourseWork3M
Курсовая работа по МОБС, 3 семестр магистратуры
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
!pip install EMD-signal

#для графиков
%matplotlib inline
 
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 12

import pandas as pd

data = pd.read_csv('/Users/lidia/Desktop/Магистратура/Кобелев/Курсовая 3 сем/2025-03-20 18-17-32 266 rcms.csv')
data.head()

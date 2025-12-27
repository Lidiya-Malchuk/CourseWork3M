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

# 1-й столбец — время
t = data.iloc[:, 0].values

# 3-й столбец — базовый импеданс
z0 = data.iloc[:, 2].values

print(len(t), len(z0))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(t, z0)
plt.title('Исходный сигнал базового импеданса')
plt.xlabel('Время, с')
plt.ylabel('Импеданс, Ом')
plt.grid()
plt.show()

#Частота дискретизации
import numpy as np

# Разность соседних временных отсчётов
dt = np.diff(t)

# Средний шаг по времени
dt_mean = np.mean(dt)

# Частота дискретизации
fs = 1 / dt_mean

print(f"Средний шаг по времени dt = {dt_mean:.6f} с")
print(f"Частота дискретизации fs = {fs:.2f} Гц")

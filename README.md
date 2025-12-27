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

from scipy.signal import detrend

z0_proc = detrend(z0)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(t, z0_proc, linewidth=1.5)
plt.xlabel('Время, с')
plt.ylabel('Импеданс, Ом')
plt.title('Сигнал базового импеданса после удаления тренда')
plt.grid()

plt.show()

#проектирование фильтра
from scipy.signal import butter, filtfilt

fc = 0.5  # Гц — выше дыхания
wn = fc / (fs / 2)

b, a = butter(N=4, Wn=wn, btype='low')

#применение фильтра
z0_filt = filtfilt(b, a, z0_proc)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(t, z0, label='Исходный Z₀', alpha=0.6)
plt.plot(t, z0_filt, label='После предобработки', linewidth=2)
plt.xlabel('Время, с')
plt.ylabel('Импеданс, Ом')
plt.title('Предобработка базового импеданса')
plt.legend()
plt.grid()
plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(t, z0_filt, linewidth=1.5)
plt.xlabel('Время, с')
plt.ylabel('Импеданс, Ом')
plt.title('Предобработка базового импеданса')
plt.grid()
plt.show()

# Частота дискретизации
fs = 1 / dt_mean

print(f"Средний шаг по времени dt = {dt_mean:.6f} с")
print(f"Частота дискретизации fs = {fs:.2f} Гц")

from PyEMD import EMD, EEMD
from scipy.signal import welch
import matplotlib.pyplot as plt
import numpy as np
# длительность окна (сек)
window_length = 60  

# интересующие моменты времени (сек)
centers_sec = [120, 960, 2100]  # 2, 16, 35 мин

segments = []  # список для хранения фрагментов

for center in centers_sec:
    # логическая маска по времени
    mask = (t >= center) & (t < center + window_length)
    
    t_seg = t[mask]
    z_seg = z0_filt[mask]
    
    segments.append((t_seg, z_seg))
    
    print(f"Фрагмент {center//60}-я минута: "
          f"{t_seg[0]:.1f}–{t_seg[-1]:.1f} c, "
          f"{len(z_seg)} отсчётов")

print(f"Начало записи: t_min = {t.min():.2f} c")
print(f"Конец записи:  t_max = {t.max():.2f} c")
print(f"Длительность записи: {(t.max() - t.min())/60:.2f} мин")

# длительность окна (сек)
window_length = 60  

# интересующие моменты времени (сек)
centers_sec = [120, 960, 1920]  # 2, 16, 32 мин

segments = []  # список для хранения фрагментов

for center in centers_sec:
    # логическая маска по времени
    mask = (t >= center) & (t < center + window_length)
    
    t_seg = t[mask]
    z_seg = z0_filt[mask]
    
    segments.append((t_seg, z_seg))
    
    print(f"Фрагмент {center//60}-я минута: "
          f"{t_seg[0]:.1f}–{t_seg[-1]:.1f} c, "
          f"{len(z_seg)} отсчётов")

          # длительность окна (сек)
window_length = 60  

# интересующие моменты времени (сек)
centers_sec = [120, 960, 1860]  # 2, 16, 31 мин

segments = []  # список для хранения фрагментов

for center in centers_sec:
    # логическая маска по времени
    mask = (t >= center) & (t < center + window_length)
    
    t_seg = t[mask]
    z_seg = z0_filt[mask]
    
    segments.append((t_seg, z_seg))
    
    print(f"Фрагмент {center//60}-я минута: "
          f"{t_seg[0]:.1f}–{t_seg[-1]:.1f} c, "
          f"{len(z_seg)} отсчётов")
          plt.figure(figsize=(12, 5))

for i, (t_seg, z_seg) in enumerate(segments):
    plt.plot(t_seg - t_seg[0], z_seg, label=f'{centers_sec[i]//60}-я минута')

plt.xlabel('Время внутри окна, с')
plt.ylabel('Импеданс, Ом')
plt.title('Выбранные 60-секундные фрагменты базового импеданса')
plt.legend()
plt.grid()
plt.show()

#график 2-х участков сигнала
plt.figure(figsize=(15, 7))

for i, (t_seg, z_seg) in enumerate(segments):
    plt.plot(t_seg - t_seg[0], z_seg, label=f'промежуток {centers_sec[i]//120}-{centers_sec[i]//120 + 1} минут')

plt.xlabel('Время внутри окна, с')
plt.ylabel('Импеданс, Ом')
plt.title('Выбранные 120-секундные фрагменты базового импеданса')
plt.legend()
plt.grid()
plt.show()

# берём окно 1-2 минут
t_seg, z_seg = segments[0]

# сделаем локальную временную ось (0-120.0 с)
t_local = t_seg - t_seg[0]

plt.figure(figsize=(10, 4))
plt.plot(t_local, z_seg)
plt.xlabel('Время, с')
plt.ylabel('Импеданс, Ом')
plt.title('Окно 2-й минуты (сигнал перед EEMD)')
plt.grid()
plt.show()

from PyEMD import EEMD
eemd = EEMD()

# стандартное отклонение сигнала
signal_std = np.std(z_seg)

eemd.noise_width = 0.1 * signal_std
eemd.trials = 50

print(f"noise_width = {eemd.noise_width:.4f}")
print(f"trials = {eemd.trials}")
# запуск EEMD
IMFs = eemd.eemd(z_seg, t_local)

print(f"Количество IMF: {IMFs.shape[0]}")

num_imfs = IMFs.shape[0]

plt.figure(figsize=(12, 2 * num_imfs))

for i in range(num_imfs):
    plt.subplot(num_imfs, 1, i + 1)
    plt.plot(t_local, IMFs[i])
    plt.ylabel(f'IMF {i+1}')
    plt.grid()
    
plt.xlabel('Время, с')
plt.suptitle('IMF после EEMD (окно 2-й минуты)', y=0.92)
plt.show()


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

num_imfs = IMFs.shape[0]
cols = 3  # количество колонок
rows = [5, 5, 3]  # количество графиков в каждой колонке

plt.figure(figsize=(15, 10))
gs = GridSpec(max(rows), cols, hspace=0.4, wspace=0.3)

imf_idx = 0
for col in range(cols):
    for row in range(rows[col]):
        if imf_idx >= num_imfs:
            break
        ax = plt.subplot(gs[row, col])
        ax.plot(t_local, IMFs[imf_idx])
        ax.set_ylabel(f'IMF {imf_idx+1}')
        ax.grid(True)
        if row == max(rows)-1:  # подпись только у нижнего графика
            ax.set_xlabel('Время, с')
        imf_idx += 1

plt.suptitle('IMF после EEMD (окно 1-2 минут)', y=0.95)
plt.show()

#FFT одной IMF
from scipy.fft import rfft, rfftfreq

# номер IMF для примера
imf_idx = 0

imf = IMFs[imf_idx]

# FFT
fft_vals = np.abs(rfft(imf))
freqs = rfftfreq(len(imf), d=1/fs)

plt.figure(figsize=(10, 4))
plt.plot(freqs, fft_vals)
plt.xlim(0, 2)  # интересуют низкие частоты
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда')
plt.title(f'Спектр IMF {imf_idx+1}')
plt.grid()
plt.show()


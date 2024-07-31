#Works for the WheelBEC setup on Nathan's table. NOT generalisable! Do not use unless you know what you're doing.

import numpy as np
import pickle
from scipy.interpolate import CubicSpline
# path = r'C:\Semiconductor Cavity\2022\202205\calib_data.pkl'
#
# # #Converts powers to angles
# # def fit_func(pwr, A, b, c, d):
# #     return A*(np.arccos(np.sqrt(pwr) * b - c))**2 + d
# #
# # with open(path, 'rb') as f:
# #     angle_list, power_list = pickle.load(f)
# #
# #

import sys
import time
import numpy as np
import pylab as plt
import socket
import thorlabs_apt as apt
import pyvisa as visa
from ThorlabsPM100 import ThorlabsPM100
from tqdm import tqdm
import pickle

power_meter_usb_name = 'USB0::0x1313::0x8078::P0034379::INSTR'

power_meter = ThorlabsPM100(visa.ResourceManager().open_resource(power_meter_usb_name, timeout=10))
power_meter.sense.power.dc.range.auto = "OFF"
power_meter.configure.scalar.power()
motor = apt.Motor(83854619)

angle_start = 40
motor.move_to(0)
time.sleep(15)
angle_stop = 90
step = 3
angle_list = np.arange(angle_start, angle_stop + step, step)
angle_list_radians = np.array(angle_list) * np.pi / 180

power_list = list()

for angle in tqdm(angle_list):
    motor.move_to(angle)
    time.sleep(3)
    #temppowerlist = list()
    #for n in range(100):
    #    temppowerlist.append(power_meter.read)
    print(power_meter.fetch)
    power_list.append(power_meter.read/0.3)

motor.move_to(angle_list[0])



print(angle_list)
print(power_list)
np.savetxt('angle_list.txt', angle_list)
np.savetxt('power_list.txt', power_list)


plt.plot(angle_list, power_list)
plt.show()

# angle_list, power_list = (angle_list[len(angle_list)], power_list[len(angle_list)//2:])
#%%
angle_list = angle_list[1:-1]
power_list = power_list[1:-1]


phi = np.linspace(power_list[0], power_list[-1],100)
pwr_toangle = CubicSpline(power_list, angle_list)
plt.plot(power_list, angle_list, '.',  label = 'data')
plt.plot(phi, pwr_toangle(phi), label = 'Interpolation')
plt.legend()
plt.show()

with open('pwr_toangle.pkl', 'wb') as f:
    pickle.dump((pwr_toangle, power_list[0], power_list[-1]), f)


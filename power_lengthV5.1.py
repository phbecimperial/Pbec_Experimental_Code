import numpy as np
from tqdm import tqdm
import Components as comp
from Measure import power_scan
import winsound
import time

components = dict()

components.update({"laser": comp.HWP_Laser()})
components.update({"lasermeter": comp.Thor_PowerMeter(num_power_readings=100, bs_factor=3/7, laser=True, wavelength=784)})
components.update({"powermeter": comp.Thor_PowerMeter(power_meter_usb_name='USB0::0x1313::0x8078::P0046773::INSTR', num_power_readings=100, bs_factor=0.25, wavelength=950)})
components.update({"spectrometer": comp.Spectrometer(spec_nd=1 / 40000)})
components.update({"wheel": comp.FilterWheel(com_port='COM9')})
components.update({"camera": comp.Camera(measure=True)})


#Define PCA range
length_list = np.linspace(-5, 8, 8)
#Define power range
power_list = np.linspace(components['laser'].pmin, components['laser'].pmax, 15)


time_stamps = []
print('Starting')

components['laser'].set_power(components['laser'].pmin)
time.sleep(5)


for l in tqdm(length_list, leave=True):
    comp.set_lock(l) #save pca value
    ts = power_scan(power_list, components, l)
    #Append and save timestamps
    time_stamps.append(ts)
    components['laser'].set_power(components['laser'].pmin)
    time.sleep(10)


winsound.Beep(1000, 2000)
print("'"+str(time_stamps[0][0])+"'"+","+"'"+str(time_stamps[-1][-1])+"'")




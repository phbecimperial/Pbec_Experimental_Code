import numpy as np
from tqdm import tqdm
import Components as comp
from Measure import power_scan
import winsound

components = dict()

components.update({"powermeter": comp.PowerMeter(num_power_readings=100, bs_factor=1/4)})
components.update({"spectrometer": comp.Spectrometer(spec_nd=1 / 40000)})
components.update({"wheel": comp.FilterWheel(com_port='COM9')})
components.update({"camera": comp.Camera(measure=True)})
components.update({"laser": comp.HWP_Laser()})


#Define PCA range
length_list = np.linspace(5, -2, 12)
#Define power range
power_list = np.linspace(components['laser'].pmin, components['laser'].pmax, 25)


time_stamps = []
print('Starting')
for l in tqdm(length_list, leave=True):
    comp.set_lock(l) #save pca value
    ts = power_scan(power_list, components, l)
    #Append and save timestamps
    time_stamps.append(ts)

power_scan(power_list, components)

winsound.Beep(100, 1000)
print("'"+str(time_stamps[0])+"'"+","+"'"+str(time_stamps[-1])+"'")




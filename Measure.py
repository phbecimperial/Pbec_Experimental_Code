import time
import logging
import Components as comp
import socket
import sys
import numpy as np
import pbec_ipc
from tqdm.autonotebook import tqdm
import warnings
import re

if socket.gethostname() == "ph-photonbec5":
    sys.path.append(r"D:/Control/PythonPackages/")

from pbec_analysis import make_timestamp, ExperimentalDataSet


class Measure():
    def __init__(self, comps, power = None, PCA=np.nan, position = None): #comps is a dictionary
        self.comps = comps
        self.power =  power
        self.position = position
        #self.laser = comps['laser']
        self.timestamp = make_timestamp(precision=0)
        self.dataset = ExperimentalDataSet(self.timestamp)
        self.PCA=PCA #Optional
        comp.params.update({'pca': self.PCA})

    def powermeter(self):
        self.comps['powermeter'].take_power_reading()

    def lasermeter(self):
        self.comps['lasermeter'].take_power_reading()
        #print('lasermeter complete')

    def spectrometer(self):
        spec = self.comps['spectrometer']
        integration_time = spec.initial_time

        spec.get_spectrometer_data(integration_time, spec.total_time)

        # repeats if max of spectrum above max allowed count rate
        while spec.saturated and integration_time >= spec.min_int_time:

            # divides integration time by 5, unless if below min count rate
            integration_time = max([integration_time / 5, spec.min_int_time])
            spec.get_spectrometer_data(integration_time, spec.total_time)

            # if intensity still too large, move to next filter
            if spec.saturated and round(integration_time, 2) <= 2 * round(spec.min_int_time, 2):
                self.comps['wheel'].increase_filter()
                integration_time = spec.initial_time
                spec.get_spectrometer_data(integration_time, spec.total_time)

        spec.get_spectrometer_data(integration_time, spec.total_time)

        # Save spectrometer data to custom dataset, details in pbec analysis
        spec.get_cavity_length()
        spec.save_reset(self.dataset, self.timestamp)

    def camera(self):
        cam = self.comps['camera']
        cam.take_pic()
        while cam.cam_saturated and cam.exposure == cam.min_exposure:
            self.comps['wheel'].increase_filter()
            time.sleep(1)
            cam.take_pic()

        try:
           cavity_length = self.comps['spectrometer'].cavity_length
        except Exception:
            cavity_length = np.nan
            warnings.warn('Spectrometer not avaliable or other error.')
        image_name = str(self.timestamp) + '_' + str(cam.exposure) + '_' + str(self.power) + '_' + str(
            cavity_length) + '_' + str(self.PCA)
        cam.save_pic(self.dataset, image_name)
        return image_name + '_.png'



    def take_measurement(self):
        time.sleep(2)

        for key, value in self.comps.items(): #Get names and component objects from dictionary
            if value.measure == True:
                #key = re.sub(r'\d', '', key)
                measure_func = getattr(self, key)
                measure_func()
                logging.info(f'{key} complete')

        comp.update_dataset(self.dataset)

        self.dataset.saveAllData()
        return str(self.timestamp), self.dataset


def threshold_scan(p_list, scale=0.2, new_points=2,
                   resolution=0.001, components=None):
    if components is None:
        raise Exception('Need components')

    timestamps = []
    input_pwrs = []
    output_pwrs = []
    output_pwrs.append(p_list.tolist())

    # Reset
    components['wheel'].reset()

    # Takes an initial measurement given by a linspace passed in by user
    for pwr in tqdm(p_list):
        # Set up measure class
        Measure_obj = Measure(components, pwr)

        # Take measurement
        timestamp = Measure_obj.take_measurement()
        timestamps.append(timestamp)  # Do not remove the spaces.
        output_pwrs.append(comp.params['meter_reading'])

    ps = input_pwrs
    intlist = np.array(output_pwrs)

    # calculates difference between log10 of intensity
    diff_list = abs(np.diff(np.log10(intlist)))
    max_diff_0 = np.argmax(diff_list)
    max_diff_i = max_diff_0

    n_plist_spacing = np.diff(ps) / new_points  # spacing between two points
    # If spacing between points below resolution,
    # set between points to diff to zero so no new measurements taken in region
    diff_list[n_plist_spacing < resolution] = 0

    iteration = 0

    while diff_list[max_diff_i] >= scale or (diff_list != 0).all():
        ps_args = np.argwhere(diff_list >= scale)
        new_ps = []
        for i, arg in enumerate(ps_args):
            print(arg)
            print(np.linspace(ps[arg], ps[arg + 1], new_points + 2)[1:-1])
            for j in np.linspace(ps[arg], ps[arg + 1], new_points + 2)[1:-1]:
                print(j)
                new_ps.append(j[0])

        components['wheel'].reset()
        ps.append(new_ps)

        for i in tqdm(new_ps):
            print(i, type(i))
            Measure_obj = Measure(components, pwr)

            # Take measurement
            timestamp = Measure_obj.take_measurement()
            timestamps.append(timestamp)  # Do not remove the spaces.
            output_pwrs.append(comp.params['meter reading'])

        if len(intlist) == 0:
            break  # So it doesn't break, temporary
        diff_list = abs(np.diff(np.log10(intlist)))
        max_diff_0 = np.argmax(diff_list)
        max_diff_i = max_diff_0

        n_plist_spacing = np.diff(ps) / new_points  # spacing between two points
        diff_list[n_plist_spacing < resolution] = 0  # If maximum

        iteration += 1

        print('\'', timestamps[0].strip(), '\',\'', timestamps[-1].strip(), '\'')

def power_scan(p_list, components, pca=np.nan):
    time_stamps = []
    components['wheel'].reset()
    for pwr in tqdm(p_list, leave=True):
        # Reset
        components['laser'].set(pwr)
        logging.info(f'PCA: {pca}')
        # Set up measure class
        measure = Measure(components, pwr, pca)
        # Take measurement
        timestamp = measure.take_measurement()
        time_stamps.append(timestamp)

    logging.info(f'{time_stamps[0], time_stamps[-1]}')
    return time_stamps

def coherence_scan(position_list, components,pca=np.nan):
    time_stamps = []
    for position in position_list:
        # Reset
        tstage = components['stage']
        tstage.set(position)
        print('Position:',tstage.get_position())
        components['wheel'].reset()
        # Set up measure class
        measure = Measure(components, position = position,PCA=pca)
        # Take measurement
        timestamp = measure.take_measurement()
        time_stamps.append(timestamp)

    print(time_stamps[0], time_stamps[-1])
    return time_stamps
import socket
import sys

if socket.gethostname() == "ph-photonbec5":
    sys.path.append("D:/Control/PythonPackages/")
    sys.path.append("D:/Control/CameraUSB3/")
    sys.path.append("D:/Control/SpectrometersV2/")
    sys.path.append("D:/Control/PiezoController/")
    sys.path.append("D:/Control/KCubeController/")
    sys.path.append(r"C:\Users\photonbec\AppData\Local\Programs\Python\Python38-32\Lib\site-packages")

import pbec_ipc
from ThorlabsPM100 import ThorlabsPM100
import pyvisa as visa
import numpy as np
import time
from pbec_analysis import make_timestamp, SpectrometerData, ExperimentalDataSet, CameraData

params = dict()


def update_dataset(dataset):
    dataset.meta.parameters.update(params)


def set_lock(PCA):
    print(PCA)
    print(pbec_ipc.PORT_NUMBERS["cavity_lock"])
    pbec_ipc.ipc_exec("setSetPoint(" + str(float(PCA)) + ")", port=pbec_ipc.PORT_NUMBERS["cavity_lock"])


class PowerMeter():
    def __init__(self, power_meter_usb_name='USB0::0x1313::0x8078::P0034379::INSTR', num_power_readings=100,
                 bs_factor=1, measure=True):
        self.power_meter_usb_name = power_meter_usb_name
        self.power_meter = ThorlabsPM100(visa.ResourceManager().open_resource(power_meter_usb_name, timeout=10))
        self.num_power_readings = num_power_readings
        self.bs_factor = bs_factor
        self.measure = measure
        print('Found power meter')
        # return self.power_meter

    def update_measurement_params(self, num_power_readings, bs_factor):
        self.num_power_readings = num_power_readings
        self.bs_factor = bs_factor

    def take_power_reading(self):
        '''

        :param num_power_readings:
        :param bs_factor: Factor light has been decreased by (e.g. due to beamsplitters) between cavity and power meter.
                            e.g. 2 50:50 beamsplitters means the bs_factor=0.25.
        :return: mean power
        '''
        ps = []
        for i in range(self.num_power_readings):
            time.sleep(0.01)
            ps.append(self.power_meter.read)
        reading = np.mean(ps) / self.bs_factor
        params.update({'meter_reading': np.mean(ps) / self.bs_factor})
        return reading


from single_spec_IPC_module import get_spectrum_measure


class Spectrometer():

    def __init__(self, spectrometer_server_port=pbec_ipc.PORT_NUMBERS["spectrometer_server V2"],
                 spectrometer_server_host='localhost', max_count_rate=10000, min_int_time=2, spec_nd=1 / 7,
                 total_time=100, initial_time=1000,
                 measure=True

                 ):
        self.spectrometer_server_port = spectrometer_server_port
        self.spectrometer_server_host = spectrometer_server_host
        self.max_count_rate = max_count_rate
        self.min_int_time = min_int_time
        self.spec_nd = spec_nd
        self.int_time = None
        self.total_time = total_time
        self.initial_time = initial_time
        self.spectrum = None  # Note attribute will not have nd correction, however saved data will!
        self.lamb = None
        self.saturated = False
        self.cavity_length = None
        self.measure = measure
        params.update({"spectrometer_nd_filter": float(spec_nd)})  # For backwards compatibility


    def get_spectrometer_data(self, int_time, total_time):
        _, _, lamb, spectrum = get_spectrum_measure(int_time=int_time, n_averages=max(1, round(total_time / int_time)),
                                                    n_measures=2,
                                                    port=self.spectrometer_server_port,
                                                    host=self.spectrometer_server_host)
        self.spectrum = np.array(spectrum)
        self.lamb = np.array(lamb)
        self.int_time = int_time
        time.sleep(0.1)

        if np.max(self.spectrum) >= self.max_count_rate:
            self.saturated = True

        params.update({'spectrometer_integration_time': int(int_time)})

        return self.lamb, self.spectrum


    def get_cavity_length(self):
        self.cavity_length = self.lamb[np.argmax(self.spectrum)]
        params.update({'cavity_length': self.cavity_length})
        print(self.cavity_length)
        return self.cavity_length


    def save_reset(self, dataset, timestamp):
        self.saturated = False  # Reset whether saturated or not
        spectrometer_data = SpectrometerData(timestamp)
        spectrometer_data.lamb = self.lamb
        spectrometer_data.spectrum = self.spectrum
        dataset.dataset["Spectrometer"] = spectrometer_data


from microscope.filterwheels.thorlabs import ThorlabsFilterWheel


class FilterWheel():
    def __init__(self, allowed_filter_positions=[0, 5], com_port='COM5'):
        self.allowed_filter_positions = allowed_filter_positions
        self.com_port = com_port
        self.filter_wheel = ThorlabsFilterWheel(com=self.com_port)

        self.filter_wheel.initialize()
        self.filter_wheel.set_position(0)
        self.current_pos_index = 0
        self.measure = False
        params.update({'nd_filter': 0})

    def increase_filter(self):

        filter_pos = self.filter_wheel.get_position()
        if filter_pos == max(self.allowed_filter_positions):
            raise Exception('Max ND filter reached')
        else:
            self.filter_wheel.set_position(self.allowed_filter_positions[self.current_pos_index + 1])
            self.current_pos_index = self.current_pos_index + 1
            time.sleep(2.5)
            params.update({'nd_filter': self.allowed_filter_positions[self.current_pos_index]})

    def reset(self):
        self.filter_wheel.set_position(0)


from CameraUSB3 import CameraUSB3


class Camera():
    def __init__(self, camera_id='nathans_dungeon_cavity_NA1', standard_exposure_time=4, measure=True):
        self.camera = CameraUSB3(verbose=True, camera_id=camera_id, timeout=1000, acquisition_mode='continuous')
        self.standard_exposure = standard_exposure_time
        self.exposure = standard_exposure_time
        self.im = None
        self.measure = measure
        # return self.camera

    def change_exposure(self, exposure_time):
        # Warning: camera will typically round some values - below 20, to the nearest 4!
        self.camera.set_exposure_time(exposure_time)
        self.exposure = self.camera.get_exposure_time()
        params.update({"camera_integration_time": int(self.exposure)})

    def take_pic(self):
        # Now take a picture

        self.change_exposure(self.standard_exposure)
        # standard_exposure_time = 500000
        self.camera.begin_acquisition()
        image = self.camera.get_image()
        while np.amax(image) < 50 and np.amax(image) != 0 and self.exposure < 500000:
            counts = np.unique(image, return_counts=True)[1]
            if self.exposure < 20:  # dumb camera things
                self.change_exposure(max(5, self.exposure * 1.8))
            else:
                self.change_exposure(self.exposure*2)
            image = self.camera.get_image()
            time.sleep(self.exposure * 1e-6)
            print(self.exposure)

        n_frames = min(50, round(500000 / self.exposure))
        print('n_frames', n_frames)
        frames = list()
        for i in range(0, n_frames):
            frames.append(self.camera.get_image())
            time.sleep(self.exposure * 1e-6)
        self.im = np.sum(frames, axis=0) / n_frames

        self.camera.end_acquisition()

        return self.im

    def save_pic(self, dataset, fname):
        camera_data = CameraData(fname, extension='_.png')
        camera_data.data = self.im
        dataset.dataset["CavityCamera"] = camera_data
        self.change_exposure(self.standard_exposure)  # Resets


from pylablib.devices import Toptica


class Laser():
    def __init__(self, com_port='COM15'):
        self.laser = Toptica.TopticaIBeam(com_port)
        self.measure = False

    def set_power(self, power):
        self.laser.set_channel_power(1, power)
        self.power = power
        params.update({'power': self.power * 1000})

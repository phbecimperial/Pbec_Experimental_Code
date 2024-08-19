import socket
import winsound
import sys

# from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

if socket.gethostname() == "ph-photonbec5":
    sys.path.append("D:/Control/PythonPackages/")
    sys.path.append("D:/Control/CameraUSB3/")
    sys.path.append("D:/Control/SpectrometersV2/")
    sys.path.append("D:/Control/PiezoController/")
    sys.path.append("D:/Control/KCubeController/")
    sys.path.append(r"C:\Users\photonbec\AppData\Local\Programs\Python\Python38-32\Lib\site-packages")
elif socket.gethostname() == "IC-5CD341DLTT":
    sys.path.append("C:/Control/PythonPackages/")
    sys.path.append("C:/Control/CameraUSB3/")
    sys.path.append("C:/Control/SpectrometersV2/")
    sys.path.append("C:/Control/PiezoController/")
    sys.path.append("C:/Control/KCubeController/")
    sys.path.append("C:/Control/CavityLock_minisetup/")
    sys.path.append(r"C:/Control/PythonPackages/")

import pbec_ipc
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
    def __init__(self, num_power_readings=100, bs_factor=1, wavelength=950, measure=True, laser=False):
        self.num_power_readings = num_power_readings
        self.bs_factor = bs_factor
        self.measure = measure
        self.laser = laser
        self.wavelength = wavelength

    def take_power_reading(self):
        raise Exception('Not implemented')



class Thor_PowerMeter(PowerMeter):


    def __init__(self, power_meter_usb_name='USB0::0x1313::0x8078::P0034379::INSTR', num_power_readings=100,
                 bs_factor=1, wavelength=950, measure=True, laser=False):
        from pylablib.devices.Thorlabs.misc import GenericPM
        super().__init__(num_power_readings, bs_factor, wavelength, measure, laser)
        self.power_meter = GenericPM(power_meter_usb_name)
        self.power_meter.set_sensor_mode('power')
        self.power_meter.set_wavelength(wavelength)
        print('Found power meter')

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
            ps.append(self.power_meter.get_power())
        reading = np.mean(ps) / self.bs_factor
        if self.laser:
            params.update({'power': reading * 1000})
            print('power reading', reading * 1000)
        else:
            params.update({'meter_reading': reading})
        return reading


from single_spec_IPC_module import get_spectrum_measure

class Grandaddy_PowerMeter(PowerMeter):
    def __init__(self, num_power_readings=100, bs_factor=1, wavelength=950, measure=True, laser=False):
        super().__init__(num_power_readings, bs_factor, wavelength, measure, laser)

        rm = visa.ResourceManager()
        res_list = rm.list_resources()#
        port = ''
        for i in res_list:
            try:
                iden = rm.open_resource(i).query('*IDN?')
            except:
                print(f'No instrument at {i}')
            else:
                if 'NewportCorp' in iden:
                    port = i
            
        if port == '':
            raise Exception('No Newport 1835-c Power meters Found')
            
        self.power_meter = rm.open_resource(i)
        self.power_meter.write(f'STSIZE {num_power_readings};MODE DCCONT;UNITS W;LAMBDA {wavelength}; SPREC 4096;SFREQ 500;BARGRAPH 1; AUTO 1')

    def take_power_reading(self):
        reading = float(self.power_meter.query('STMEAN?'))/self.bs_factor
        return reading
    
    def set_wavelength(self, wavelength: float):
        self.power_meter.write(f'LAMBDA {wavelength}')
    
    def set_buffersize(self, size: int):
        size = min([size, 100])
        self.power_meter.write(f'STSIZE {size}')



class Spectrometer():

    def __init__(self, spectrometer_server_port=pbec_ipc.PORT_NUMBERS["spectrometer_server V2"],
                 spectrometer_server_host='localhost', max_count_rate=10000, min_int_time=2, spec_nd=1 / 7,
                 total_time=100, initial_time=1000,
                 measure=True, min_lamb=910

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
        self.min_lamb = min_lamb
        params.update({"spectrometer_nd_filter": float(spec_nd)})  # For backwards compatibility
        print('Found spectrometer')


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
        else:
            self.saturated = False

        params.update({'spectrometer_integration_time': int(int_time)})

        return self.lamb, self.spectrum

    def get_cavity_length(self):
        lamb = self.lamb[self.lamb > self.min_lamb]
        spec = self.spectrum[self.lamb > self.min_lamb]
        self.cavity_length = lamb[np.argmax(spec)]
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
    def __init__(self, allowed_filter_positions=[0, 5], com_port='COM6'):
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
            winsound.Beep(1000, 1000)
        else:
            self.filter_wheel.set_position(self.allowed_filter_positions[self.current_pos_index + 1])
            self.current_pos_index = self.current_pos_index + 1
            time.sleep(2.5)
            params.update({'nd_filter': self.allowed_filter_positions[self.current_pos_index]})

    def reset(self):
        self.filter_wheel.set_position(0)
        self.current_pos_index = 0

class Camera():
    def __init__(self, min_exposure, max_exposure, measure=True, max_frames=50, algorithm = 'rising', camera_id = None, wheel = True):
        self.camera_id = camera_id
        self.measure = measure
        self.im = None
        self.camera = None
        self.min_exposure=min_exposure
        self.max_exposure=max_exposure #Set depending on the camera itself!
        self.exposure=None
        self.max_frames = max_frames
        self.cam_saturated = False
        self.algorithm = algorithm
        self.wheel = wheel


    def change_exposure(self, exposure):
        raise Exception('Not implemented')

    def get_image(self):
        raise Exception('Not implemented')

    def get_multiple_images(self, num=10):
        #Return as a list
        raise Exception('Not implemented')
    def take_pic(self):
        #Only works if camera goes between 0 and 255

        if self.algorithm == 'rising':
            self.change_exposure(self.min_exposure)
            self.exposure = self.min_exposure
            # standard_exposure_time = 500000
            # self.camera.begin_acquisition()
            time.sleep(0.1)
            image = self.get_image()
            while np.amax(image) < 50 and np.amax(image) != 0 and self.exposure < self.max_exposure:
                self.change_exposure(self.exposure * 2)
                image = self.get_image()
                time.sleep(self.exposure * 1e-6)

        elif self.algorithm == 'falling':
            # Now take a picture
            standard_exposure_time = 500000
            self.set_exposure_time(standard_exposure_time)
            time.sleep(0.5)
            exposure_time = self.get_exposure_time()

            image = self.get_image()
            while np.amax(image) > 240 and np.amax(image) != 0 and int(exposure_time) > 5:
                if exposure_time > 20:  # dumb camera things
                    self.set_exposure_time(max(5, exposure_time * 0.8))
                else:
                    self.set_exposure_time(exposure_time / 2)
                exposure_time = self.get_exposure_time()
                image = self.get_image()
                time.sleep(exposure_time * 1e-6)
                print(exposure_time)

            raise Exception('Not implemented')

        elif self.algorithm == 'middle':
            raise Exception('Not implemented')

        n_frames = min(self.max_frames, round(self.max_exposure / self.exposure))
        print('n_frames', n_frames)



        ims = self.get_multiple_images(n_frames)
        self.im = np.sum(ims, axis=0) / n_frames

        if np.amax(image) == 255:
            self.cam_saturated = True
        else:
            self.cam_saturated = False

        params.update({"camera_integration_time": int(self.exposure)})

        return self.im

    def save_pic(self, dataset, fname):
        camera_data = CameraData(fname, extension='_.png')
        camera_data.data = self.im
        dataset.dataset["CavityCamera"] = camera_data




class FLIR_Camera(Camera):
    def __init__(self, camera_id='nathans_dungeon_cavity_NA1', min_exposure=4, max_exposure=500000, measure=True, max_frames=50, algorithm='rising'):
        #Define init to turn on camera.
        from CameraUSB3 import CameraUSB3
        super().__init__(min_exposure, max_exposure, measure, max_frames, algorithm, camera_id)
        self.camera = CameraUSB3(verbose=True, camera_id=self.camera_id, timeout=1000, acquisition_mode='continuous')
        #Running in continuous mode - faster!
        self.camera.begin_acquisition()
        print('Found Cameras')

    def change_exposure(self, exposure_time):
        # Warning: camera will typically round some values - below 20, to the nearest 4!
        self.camera.set_exposure_time(exposure_time)
        self.exposure = self.camera.get_exposure_time()

    def get_exposure(self):
        return self.camera.get_exposure_time()

    def get_image(self):
        image = self.camera.get_image()
        return image
        # raise Exception('Not implemented')

    def get_multiple_images(self, num=10):
        # Return as a list
        images = []
        for k in range(num):
            images.append(self.get_image())
        return images
        # raise Exception('Not implemented')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.camera.end_acquisition()

    # def take_pic(self):
    #     # Now take a picture
    #     # algorithm, rising
    #
    #     self.change_exposure(self.standard_exposure)
    #     # standard_exposure_time = 500000
    #     self.camera.begin_acquisition()
    #     image = self.camera.get_image()
    #     while np.amax(image) < 50 and np.amax(image) != 0 and self.exposure < 500000:
    #         counts = np.unique(image, return_counts=True)[1]
    #         if self.exposure < 20:  # dumb camera things
    #             self.change_exposure(max(5, self.exposure * 1.8))
    #         else:
    #             self.change_exposure(self.exposure * 2)
    #         image = self.camera.get_image()
    #         time.sleep(self.exposure * 1e-6)
    #         # print(self.exposure)
    #
    #     n_frames = min(50, round(500000 / self.exposure))
    #     print('n_frames', n_frames)
    #     frames = list()
    #     for i in range(0, n_frames):
    #         frames.append(self.camera.get_image())
    #         time.sleep(self.exposure * 1e-6)
    #     self.im = np.sum(frames, axis=0) / n_frames
    #
    #     self.camera.end_acquisition()
    #
    #     params.update({"camera_integration_time": int(self.exposure)})
    #
    #     return self.im

    # def save_pic(self, dataset, fname):
    #     camera_data = CameraData(fname, extension='_.png')
    #     camera_data.data = self.im
    #     dataset.dataset["CavityCamera"] = camera_data
    #     self.change_exposure(self.standard_exposure)  # Resets

from PIL import Image
class Thor_Camera(Camera):

    def __init__(self,
                min_exposure = 100, max_exposure = 1999994,measure=True, max_frames=50, algorithm='rising'):
        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera
        import os
        try:
            # if on Windows, use the provided setup script to add the DLLs folder to the PATH
            from windows_setup import configure_path
            configure_path()
        except ImportError:
            configure_path = None
        os.add_dll_directory(
            r'C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific Camera Interfaces\SDK\Native Toolkit\dlls\Native_32_lib')
        super().__init__(min_exposure, max_exposure, measure, max_frames, algorithm)

        self.sdk = TLCameraSDK()
        self.tlc = TLCamera
        available_cameras = self.sdk.discover_available_cameras()
        if len(available_cameras) < 1:
            print("no cameras detected")
        self.camera = self.sdk.open_camera(available_cameras[0])
        self.tlc.arm(self.camera,50)

    def change_exposure(self, exposure_time):
        self.camera.exposure_time_us = int(exposure_time)
        self.exposure = self.camera.exposure_time_us
        # raise Exception('Not implemented')

    def get_exposure(self):
        return self.camera.exposure_time_us
    def get_image(self):
        self.camera.issue_software_trigger()
        frame = self.tlc.get_pending_frame_or_null(self.camera)
        image_data = frame.image_buffer
        image = Image.fromarray(image_data)
        image = np.array(image)
        return image
        # raise Exception('Not implemented')

    def get_multiple_images(self, num=10):
        # Return as a list
        images = []
        for k in range(num):
            images.append(self.get_image())
        return images
        # raise Exception('Not implemented')

    def begin_acquisition(self):
        self.tlc.arm(self.camera,50)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tlc.disarm(self.camera)
        self.tlc.dispose(self.camera)
        self.sdk.dispose()

    # def __del__(self, exc_type, exc_val, exc_tb):
    #     self.tlc.disarm(self.camera)
    #     self.tlc.dispose(self.camera)
    #     self.sdk.dispose()




class Laser():
    '''Example function for Laser class. Real world classes appear below.'''

    def __init__(self):
        self.measure = False

    def set_power(self, power):
        # Insert mechanism
        1 == 1
        self.power = power
        params.update({'power': self.power * 1000})


class Translation_Stage():
    def __init__(self, device_id=73852194, scale=20000, is_rack_system=True):
        from pylablib.devices import Thorlabs
        self.stage = Thorlabs.KinesisMotor(str(device_id), is_rack_system=is_rack_system, scale=scale)
        print('Stage Found, postion:', self.stage.get_position())
        self.stage.home()
        self.stage.wait_for_home()
        print("Stage is homed and operational")
        self.measure = False

    def set(self, position, timeout=1):
        self.stage.move_to(position)
        self.stage.wait_move()
        time.sleep(timeout)

    def get_position(self):
        return self.stage.get_position()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stage.close()

class Toptica_Laser():
    def __init__(self, com_port='COM15'):
        from pylablib.devices import Toptica
        self.laser = Toptica.TopticaIBeam(com_port)
        self.measure = False
        print('Found laser')

    def set(self, power):
        self.laser.set_channel_power(1, power)
        self.power = power
        params.update({'power': self.power * 1000})


import pickle



class HWP_Laser():
    def __init__(self, T_cube_no=int(83854619), path='pwr_toangle.pkl'):
        import thorlabs_apt as apt
        self.motor = apt.Motor(T_cube_no)
        with open(path, 'rb') as f:
            self.power_toangle, self.pmin, self.pmax = pickle.load(f)
        self.measure = False
        print('Found laser')

    def set(self, power):
        self.power = power
        self.motor.move_to(self.power_toangle(self.power))
        time.sleep(1)
        params.update({'power': self.power * 1000})

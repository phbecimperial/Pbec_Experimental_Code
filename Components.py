import socket
import winsound
import logging
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

#Create the metaparams dictionary
params = dict()


def update_dataset(dataset):
    """Add the current parameters dictionary to the dataset object

    """
    dataset.meta.parameters.update(params)


def set_lock(PCA):
    """Set the PCA on the cavity lock - change wavelength automatically

    Args:
        PCA (float)
    """
    logging.info(f'Set to PCA : {PCA}')
    logging.info(f"Cavity lock Port {pbec_ipc.PORT_NUMBERS['cavity_lock']}")
    pbec_ipc.ipc_exec("setSetPoint(" + str(float(PCA)) + ")", port=pbec_ipc.PORT_NUMBERS["cavity_lock"])

class PowerMeter():
    """General class for power meter. Inherit from this class. Must have these functions
    """    
    def __init__(self, num_power_readings=100, bs_factor=1, wavelength=950, measure=True, label='power'):
        """_summary_

        Args:
            num_power_readings (int, optional): Number of readings to average over. Defaults to 100.
            bs_factor (float, max = 1, optional): Fraction of output power reading power meter. Defaults to 1.
            wavelength (int, optional): Affects power reading. Defaults to 950.
            measure (bool, optional): Whether measurement should be taken. Defaults to True.
            label (str, optional): label for reading - defaults for power, but change it for multiple power meters. Defaults to False.
        """        
        self.num_power_readings = num_power_readings
        self.bs_factor = bs_factor
        self.measure = measure
        self.label = label
        self.wavelength = wavelength

    def take_power_reading(self):
        raise Exception('Not implemented')



class Thor_PowerMeter(PowerMeter):
    """
    Compatible with Thorlabs PM100D power meter. Note if not detected by pylablib, install thorlabs pm100d software and use 
    the driver switcher to switch drivers. Should be detected in NIMAX software. 
    """


    def __init__(self, power_meter_usb_name='USB0::0x1313::0x8078::P0034379::INSTR', num_power_readings=100,
                 bs_factor=1, wavelength=950, measure=True, label='power'):
        """
        Args:
            power_meter_usb_name (str, optional): Visa resource string. Look at NI Max software. Defaults to 'USB0::0x1313::0x8078::P0034379::INSTR'.
        """        
        
        from pylablib.devices.Thorlabs.misc import GenericPM
        super().__init__(num_power_readings, bs_factor, wavelength, measure, label)
        
        self.power_meter = GenericPM(power_meter_usb_name)
        self.power_meter.set_sensor_mode('power')
        self.power_meter.set_wavelength(wavelength)
        logging.info('Found power meter')

    def take_power_reading(self):
        '''
        Takes average power reading. Set parameters num_power_readings and bs_factor in init. 
        :return: mean power
        '''
        ps = []
        for i in range(self.num_power_readings):
            time.sleep(0.01)
            ps.append(self.power_meter.get_power())
        reading = np.mean(ps) / self.bs_factor
        params.update({'{}_meter_reading'.format(self.label): reading})
        logging.info('{}_meter_reading'.format(self.label), reading * 1000)
        return reading



class Grandaddy_PowerMeter(PowerMeter):
    def __init__(self, num_power_readings=100, bs_factor=1, wavelength=950, measure=True, label=False):
        """Newport 1935-C power meter. 
        This automatically finds the first newport corp device - NOT COMPATIBLE WITH OTHER NEWPORT DEVICES WHICH USE SERIAL!
        Automatically sets correct wavelength, number of items to average over and measurement frequency.
        Refer to Newport serial reference for more commands.

        Raises:
            Exception: _description_
        """        
        
        super().__init__(num_power_readings, bs_factor, wavelength, measure, label)

        rm = visa.ResourceManager()
        res_list = rm.list_resources()#
        port = ''
        for i in res_list:
            try:
                iden = rm.open_resource(i).query('*IDN?')
            except:
                logging.error(f'No instrument at {i}')
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


class GenericSpectrometer():
    """Base spectrometer class
    """
    def __init__(self, max_count_rate=10000, spec_nd=1/7, measure=True, min_lamb=910):
        """Sets up some standard parameters, used for any spectrometer 

        Args:
            max_count_rate (int, optional): Saturation threshold. DEPENDS ON SPECTROMETER!
            spec_nd (float, optional): Fraction of light reaching spectrometer, where reduction is due to ND filter placed 
            directly in front of spectrometer, usually to stop saturation. Does not take into account filter wheel. Defaults to 1/7.
            measure (bool, optional): Whether measurement should be taken. Defaults to True.
            min_lamb (float, optional): Wavelengths below the minimum are removed. Defaults to 910.
        """    
        params.update({"spectrometer_nd_filter": float(spec_nd)})  
        self.measure=measure
        self.cavity_length=None
        self.min_lamb=min_lamb
        self.spec_nd=spec_nd
        self.spectrum = None  
        self.lamb = None
        self.saturated = False
        self.max_count_rate = max_count_rate
    
    def get_spectrometer_data(int_time, total_time):
        """Returns an averaged spectrum, saved to object attributes self.lamb and self.spectrum.
        int_time tells you integration time of each measurement (to stop saturation). total_time/int_time tells you number
        of measurements averaged over. Set initial time (when creating the object) as higher to allow for more dynamic range.

        Also, checks whether spectrometer is saturated, and saves most recent integration time!

        Args:
            int_time (int): Integration time for each measurement
            total_time (int): Maximum total measurement time

        Raises:
            Exception: _description_

        Returns:
            numpy array: wavelength and associated spectrum
        """    
        #Should return wavelength and spectrum
        raise Exception('Not implemented Error')

        if np.max(self.spectrum) >= self.max_count_rate:
            self.saturated = True
        else:
            self.saturated = False
        
        params.update({'spectrometer_integration_time': int(int_time)})

        return self.lamb, self.spectrum

    
    def get_cavity_length(self):
        """
        Finds the peak wavelength emitted from the cavity, by taking the max of the most recent spectrum, filtering wavelengths before min_lamb.
        Saved in meta. 

        Returns:
            float: Cavity length
        """    
        lamb = self.lamb[self.lamb > self.min_lamb]
        spec = self.spectrum[self.lamb > self.min_lamb]
        self.cavity_length = lamb[np.argmax(spec)]
        params.update({'cavity_length': self.cavity_length})
        logging.info(f'Getting cavity length: {self.cavity_length}')
        return self.cavity_length

    def save_reset(self, dataset, timestamp):
        """Saves the most recent spectra to a json, and sets the spectrometer to unsaturated

        Args:
            dataset (custom): dataset object, from pbec_analysis
            timestamp (custom): from pbec_analysis 'make_timestamp'
        """    
        self.saturated = False  # Reset whether saturated or not
        spectrometer_data = SpectrometerData(timestamp)
        spectrometer_data.lamb = self.lamb
        spectrometer_data.spectrum = self.spectrum
        dataset.dataset["Spectrometer"] = spectrometer_data

class Spectrometer(GenericSpectrometer):
    """Spectrometer class compatible with avasoft spectrometers, if single spec server file is present.
    """

    def __init__(self, spectrometer_server_port=pbec_ipc.PORT_NUMBERS["spectrometer_server V2"],
                 spectrometer_server_host='localhost', max_count_rate=10000, min_int_time=2, spec_nd=1 / 7,
                 total_time=100, initial_time=1000,
                 measure=True, min_lamb=910

                 ):
        """Avasoft Spectrometer initialisation. Requrires single_spec_server_main to be running, and using the same port numbers to allow communiation to the server.

        Args:
            spectrometer_server_port (int, optional): Check pbec_ipc port. Defaults to pbec_ipc.PORT_NUMBERS["spectrometer_server V2"].
            spectrometer_server_host (str, optional): _description_. Defaults to 'localhost'.
            max_count_rate (int, optional): Saturation Threshold. Defaults to 10000.
            min_int_time (int, optional): Minimum integration time, set by spectrometer. Defaults to 2.
            total_time (int, optional): Maximum time readings should be taken. Defaults to 100.
            initial_time (int, optional): Initial integration time. Note can be greater than total time, allowing for collection of low intensity of data, whilst limiting 
            the number of measurements for high intensity data, resulting in a speedup. Defaults to 1000.
        """                 
        super.__init__(max_count_rate, spec_nd, measure, min_lamb)
        
        from single_spec_IPC_module import get_spectrum_measure
        self.spectrometer_server_port = spectrometer_server_port
        self.spectrometer_server_host = spectrometer_server_host
        self.min_int_time = min_int_time
        self.int_time = None
        self.total_time = total_time
        self.initial_time = initial_time
        logging.info('Found spectrometer')


    def get_spectrometer_data(self, int_time, total_time):
    #Gets wavelength and spectrum for avasoft, puts it into numpy arrays and checks whether saturated, as required.
    #Also updates spectrometer integration time!
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



from microscope.filterwheels.thorlabs import ThorlabsFilterWheel


class FilterWheel():
    """Controls Thorlab filter wheel. Can set allowed wheel positions, and tracks the current filter position automatically.
    Raises an exception if you attempt to increase the wheel past its maximum
    """
    def __init__(self, allowed_filter_positions=[0,5], com_port='COM6'):
        """

        Args:
            allowed_filter_positions (list, optional): Allowed filter wheel positions. Defaults to [0,5].
            com_port (str, optional): Defaults to 'COM6'.
        """    
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
            winsound.Beep(1000, 1000)
            raise Exception('Max ND filter reached')
        else:
            self.filter_wheel.set_position(self.allowed_filter_positions[self.current_pos_index + 1])
            self.current_pos_index = self.current_pos_index + 1
            time.sleep(5)
            params.update({'nd_filter': self.allowed_filter_positions[self.current_pos_index]})

    def decrease_filter(self):
        filter_pos = self.filter_wheel.get_position()
        if filter_pos == min(self.allowed_filter_positions):
            raise Exception('Min ND filter reached')
            winsound.Beep(1000, 1000)
        else:
            self.filter_wheel.set_position(self.allowed_filter_positions[self.current_pos_index - 1])
            self.current_pos_index = self.current_pos_index - 1
            time.sleep(5)
            params.update({'nd_filter': self.allowed_filter_positions[self.current_pos_index]})

    def reset(self):
        self.filter_wheel.set_position(0)
        self.current_pos_index = 0

class Camera():
    """Base Camera class. Not usuable correctly, create a new class which inherits from this to use cameras. Contains 
    exposure time finding algorithms in 'take_pic()'. CAMERA MUST RETURN IMAGES IN 8 BIT FORMAT (0-255). 
    Cannot take pictures, create inheritor class which creates methods 'change_exposure', 'get_image'.

    Recommend use continuous image mode on camera - is faster. 
    """
    def __init__(self, min_exposure, max_exposure, measure=True, max_frames=50, algorithm = 'rising', camera_id = None, wheel = True):
        """Parameters required for automatic exposure time finder

        Args:
            min_exposure (int/float, depends on camera type): Minimum exposure time
            max_exposure (int/float, depends on camera): Maximum exposure time. 
            measure (bool, optional): _description_. Defaults to True.
            max_frames (int, optional): Maximum number of images to average over (calculates number of images by doing 
            max_exposure/min_exposure, but is limited at 50). Defaults to 50.
            algorithm ('rising' or 'falling'): Exposure time finding algorithm. Defaults to 'rising'.
            camera_id (_type_, optional): Something to identify the camera. Defaults to None.
        """    
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
        #Changes the exposure time
        raise Exception('Not implemented')

    def get_image(self):
        """Creates one image, returned as a 2D numpy array in 8 bit pixel format (range 0-255)
        """
        raise Exception('Not implemented')

    def get_multiple_images(self, num=10):
        # Return as a list
        images = []
        for k in range(num):
            images.append(self.get_image())
        return images


    def take_pic(self):
        """Contains algorithms 'rising' and 'falling' to find the best exposure time which doesn't saturate the image taken. 
        Combine with the ND filter to gain even more dynamic range (if the camera is still saturated).
        Applies to all cameras. 

        Returns:
            _type_: _description_
        """    
        #Only works if camera goes between 0 and 255

        if self.algorithm == 'rising':
            self.change_exposure(self.min_exposure)
            self.exposure = self.min_exposure
            time.sleep(0.1)
            image = self.get_image()
            while np.amax(image) < 50 and np.amax(image) != 0 and self.exposure < self.max_exposure: #50 shown to be good value to limit at for nice pics by trial and error.
                self.change_exposure(self.exposure * 2)
                image = self.get_image()

                logging.info(f'Image Exposure: {self.exposure}')

                time.sleep(self.exposure * 1e-6)

        elif self.algorithm == 'falling':
            # Now take a picture
            standard_exposure_time = 500000
            self.set_exposure_time(standard_exposure_time)
            time.sleep(0.5)
            exposure_time = self.get_exposure_time()

            image = self.get_image()
            while np.amax(image) > 240 and np.amax(image) != 0 and int(exposure_time) > 5:
                if exposure_time > 20:  # dumb camera things, cameras may round at very low exposure times. 
                    self.set_exposure_time(max(5, exposure_time * 0.8))
                else:
                    self.set_exposure_time(exposure_time / 2)
                exposure_time = self.get_exposure_time()
                image = self.get_image()
                time.sleep(exposure_time * 1e-6)
                # logging.info(f' exposure time{exposure_time}')

            raise Exception('Not implemented')

        elif self.algorithm == 'middle':
            raise Exception('Not implemented')

        n_frames = min(self.max_frames, round(self.max_exposure / self.exposure))
        logging.info(f'Taking picture with {n_frames} frames')



        ims = self.get_multiple_images(n_frames)
        self.im = np.sum(ims, axis=0) / n_frames
        # print('Max pixel: ', np.amax(self.im))
        if np.amax(self.im) == 255:
            self.cam_saturated = True
        else:
            self.cam_saturated = False

        params.update({"camera_integration_time": int(self.exposure)})

        return self.im

    def save_pic(self, dataset, fname):
        """Saves picture using custom dataset saving software

        Args:
            dataset (Custom dataset from experimental dataset file): _description_
            fname (file name): Set in measure.py
        """    
        camera_data = CameraData(fname, extension='_.png')
        camera_data.data = self.im
        dataset.dataset["CavityCamera"] = camera_data

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        #Camera shutdown procedure
        raise Exception('Camera class should not be directly used')




class FLIR_Camera(Camera):
    """Class compatible with Blackfly FLIR cameras. 

    """
    def __init__(self, camera_id='nathans_dungeon_cavity_NA1', min_exposure=4, max_exposure=500000, measure=True, max_frames=50, algorithm='rising'):
        """Note camera_ids and methods to take pictures stored inside CameraUSB3

        Args:
            min_exposure (int/float, depends on camera type): Minimum exposure time
            max_exposure (int/float, depends on camera): Maximum exposure time. 
            measure (bool, optional): _description_. Defaults to True.
            max_frames (int, optional): Maximum number of images to average over (calculates number of images by doing 
            max_exposure/min_exposure, but is limited at 50). Defaults to 50.
            algorithm ('rising' or 'falling'): Exposure time finding algorithm. Defaults to 'rising'.
            camera_id (_type_, optional): Something to identify the camera. Defaults to None.
        """
        
        #Define init to turn on camera.
        from CameraUSB3 import CameraUSB3
        super().__init__(min_exposure, max_exposure, measure, max_frames, algorithm, camera_id)
        self.camera = CameraUSB3(verbose=True, camera_id=self.camera_id, timeout=1000, acquisition_mode='continuous')
        #Running in continuous mode - faster!
        self.camera.begin_acquisition()
        logging.info('Found Cameras')

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


class Thor_Camera(Camera):
    """Class compatible with Thorlabs cameras. Care was taken to close the camera. Need to install thorcam, and thorlabs python sdk - google for instructions.
    """
    def __init__(self,
                min_exposure = 100, max_exposure = 1999994,measure=True, max_frames=50, algorithm='rising'):
        """Initialisation of Thorcam.

        Args:
            min_exposure (int/float, depends on camera type): Minimum exposure time
            max_exposure (int/float, depends on camera): Maximum exposure time. 
            measure (bool, optional): _description_. Defaults to True.
            max_frames (int, optional): Maximum number of images to average over (calculates number of images by doing 
            max_exposure/min_exposure, but is limited at 50). Defaults to 50.
            algorithm ('rising' or 'falling'): Exposure time finding algorithm. Defaults to 'rising'.
        """
        
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
            logging.error("no cameras detected")
        self.camera = self.sdk.open_camera(available_cameras[0])
        self.tlc.arm(self.camera,50)

    def change_exposure(self, exposure_time):
        self.camera.exposure_time_us = int(exposure_time)
        self.exposure = self.camera.exposure_time_us

    def get_exposure(self):
        return self.camera.exposure_time_us
    
    def get_image(self):
        self.camera.issue_software_trigger()
        frame = self.tlc.get_pending_frame_or_null(self.camera)
        image_data = frame.image_buffer

        return image_data*(255/1023)

    def get_multiple_images(self, num=10):
        # Return as a list
        images = []
        for k in range(num):
            images.append(self.get_image())
        return images

    def begin_acquisition(self):
        self.tlc.arm(self.camera,50)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        #Close the camera, then close the sdk 
        self.tlc.disarm(self.camera)
        self.tlc.dispose(self.camera)
        self.sdk.dispose()


class Laser():
    '''Example function for Laser class. Real world classes appear below.'''

    def __init__(self):
        '''Lasers cannot collect data'''
        self.measure = False

    def set(self, power):
        '''Generic set method'''
        # Insert mechanism
        1 == 1
        self.power = power
        params.update({'power': self.power * 1000})


class Translation_Stage():
    """Thorlabs translation stage - works with any Thorlabs Kinesis stepper motor. 
    """    
    def __init__(self, device_id=73852194, scale=20000, is_rack_system=True):
        """_summary_

        Args:
            device_id (int, optional): . Defaults to 73852194.
            scale (int, optional): Conversion to steps. Defaults to 20000.
            is_rack_system (bool, optional): Most systems are not rack systems. See if it works. Defaults to True.
        """        
        from pylablib.devices import Thorlabs
        self.stage = Thorlabs.KinesisMotor(str(device_id), is_rack_system=is_rack_system, scale=scale)
        logging.info(f'Stage Found, postion: {self.stage.get_position()}')
        self.stage.home()
        self.stage.wait_for_home()
        logging.info("Stage is homed and operational")
        self.measure = False

    def set(self, position, timeout=1):
        self.stage.move_to(position)
        self.stage.wait_move()
        time.sleep(timeout)
        params.update({'position': self.stage.get_position()})

    def set_home(self):
        self.stage.home()
        self.stage.wait_for_home()
        logging.info("Stage is homed")

    def get_position(self):
        """Often more trustworthy to ask the stage on current position instead of remembering most recent position
        """
        return self.stage.get_position()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stage.close()

class Toptica_Laser():
    '''Laser class compatible with Toptica_Laser'''
    def __init__(self, com_port='COM15'):
        """

        Args:
            com_port (str, optional): Defaults to 'COM15'.
        """
        from pylablib.devices import Toptica
        self.laser = Toptica.TopticaIBeam(com_port)
        self.measure = False
        logging.info('Found laser')

    def set(self, power):
        """Set and saves laser power.
        """
        self.laser.set_channel_power(1, power)
        self.power = power
        params.update({'power': self.power * 1000})


import pickle

class HWP_Laser():
    #Use power_calibv2 to create pwr_toangle.pkl, which converts powers to angles
    def __init__(self, T_cube_no=int(83854619), path='pwr_toangle.pkl'):
        import thorlabs_apt as apt
        self.motor = apt.Motor(T_cube_no)
        with open(path, 'rb') as f:
            self.power_toangle, self.pmin, self.pmax = pickle.load(f)
        self.measure = False
        logging.info('Found HWP motor')

    def set(self, power):
        self.power = power
        self.motor.move_to(self.power_toangle(self.power))
        time.sleep(1)
        params.update({'power': self.power * 1000})

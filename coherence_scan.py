import sys
import time
import numpy as np
import socket
import os


if socket.gethostname() == "ph-photonbec5":
    sys.path.append("D:/Control/PythonPackages/")
    sys.path.append("D:/Control/CameraUSB3/")
    sys.path.append("D:/Control/SpectrometersV2/")
    sys.path.append("D:/Control/PiezoController/")
    sys.path.append("D:/Control/KCubeController/")

from pbec_analysis import make_timestamp,ExperimentalDataSet, CameraData

# ____________________________________________FILTER WHEEL_____________________________________________

from microscope.filterwheels.thorlabs import ThorlabsFilterWheel

# PARAMETERS
allowed_filter_positions = [0, 1, 2, 3, 5]
filter_wheel = ThorlabsFilterWheel(com='COM6')
filter_wheel.initialize()
filter_wheel.set_position(0)
nd_on = False
filter_pos = 0

# # _________________________________________TRANSLATION STAGE__________________________________________
from pylablib.devices import Thorlabs
stage = Thorlabs.KinesisMotor('73852194', is_rack_system=True, scale=20000)
print(stage.get_position())

# PARAMETERS
position_list_one =np.array([106, 106.25, 106.5, 107, 107.5, 108, 108.5, 109, 109.5, 110, 111, 112, 113, 114, 115])
position_list_two =np.array([106, 108, 110, 112, 115, 118, 120, 130, 140, 150, 160, 170, 180, 190, 200])
position_list_three =np.array([106, 106.25, 106.5, 107, 107.5, 108, 108.5, 109, 109.5, 110, 111, 112, 113, 114, 115, 118, 120, 130, 140, 150, 160,
     170, 180, 190, 200])
position_list_four =np.array([221, 220.5, 220, 219.5,219,218.5,218,217.5,217,216.5,216,215.5,215,214.5,214,213.5,213,212.5,212,211.5,211,210.5,210,209.5,209,208.5,208,207.5,207,206.5,206,205.5,205])
position_list_five = np.arange(221,90,-1)

list_to_investigate = position_list_five

# Initialisation

stage.home()
stage.wait_for_home()
print("Stage is homed and operational")

#______________________________________________CAMERA_________________________________________________
# from CameraUSB3 import CameraUSB3
#
# camera_id = 'blackfly_michelson'
# camera = CameraUSB3(verbose=True, camera_id=camera_id, timeout=1000, acquisition_mode='single frame')
#
# # PARAMETERS
# standard_exposure_time = 22
# min_exp_time = 38
# max_exp_time = 23872
# camera.set_exposure_time(standard_exposure_time)
# exposure_time = camera.get_exposure_time()
#
# # INITIAL EXPOSURE TIME SETTING
#
# image = camera.get_image()
# max_pixel = np.amax(image)

import Components as comp
import matplotlib.pyplot as plt

standard_exposure_time = 2000
min_exp_time = 100
max_exp_time = 1999994

# if max_pixel > 250 : # First exposure check
#     while np.amax(image) > 250:
#         camera.set_exposure_time(max(min_exp_time, int(exposure_time / 1.2)))
#         exposure_time = camera.get_exposure_time()
#         image = camera.get_image()
#         max_pixel = np.amax(image)
#         flag += 1
#         print(flag,exposure_time, print(int(max_pixel)))
#         if exposure_time == min_exp_time:
#             print('need nd')
#             filter_pos_idx = allowed_filter_positions.index(filter_pos)
#             filter_pos = allowed_filter_positions[filter_pos_idx + 1]
#             filter_wheel.set_position(filter_pos)
#             print('nd pos ' + str(filter_pos))
#             camera.set_exposure_time(standard_exposure_time)
#             exposure_time = camera.get_exposure_time()
#             time.sleep(2)
# #
#

# #______________________________________________COMPUTATION_________________________________________________

print('Starting measurement')

stage.move_to(list_to_investigate[0])
time.sleep(5)
real_position = []

for position in list_to_investigate:
    stage.move_to(position)
    stage.wait_move()
    time.sleep(0.3)

    # EXPOSURE ADJUSTMENT
    test = camera.get_image()
    max_pixel = np.amax(test)
    flag = 0
    fleg = 0

    if max_pixel > 250:  # First exposure check
        while np.amax(image) > 250:
            camera.set_exposure_time(max(min_exp_time, exposure_time / 1.2))
            exposure_time = camera.get_exposure_time()
            image = camera.get_image()
            max_pixel = np.amax(image)
            flag += 1
            print('Decreasing exposure time, iteration no.: ',flag, 'New exposure is: ', exposure_time,
                  'Max pixel value :', (int(max_pixel)))
            if int(exposure_time) == min_exp_time:
                print('need nd')
                filter_pos_idx = allowed_filter_positions.index(filter_pos)
                filter_pos = allowed_filter_positions[filter_pos_idx + 1]
                filter_wheel.set_position(filter_pos)
                print('nd pos ' + str(filter_pos))
                camera.set_exposure_time(max(min_exp_time, int(exposure_time / 1.2)))
                exposure_time = camera.get_exposure_time()
                time.sleep(2)
        print('Snap after decreasing exposure')
        # image = camera.get_image()
        # timestamp = make_timestamp(precision=0)
        # dataset = ExperimentalDataSet(timestamp)
        # camera_data = CameraData(timestamp,
        #                          extension='_' + str(position) + '_' + str(int(exposure_time)) + '_single' + '.png')
        # camera_data.data = image
        # dataset.dataset["CavityCamera" + str(int(stage.get_position())) + 'single'] = camera_data
        # dataset.saveAllData()
        # print(stage.get_position())
        # real_position.append(stage.get_position())
        # time.sleep(0.2)

    if max_pixel < 200:
        while max_pixel < 200:
            camera.set_exposure_time(min(max_exp_time, exposure_time * 1.8))
            exposure_time = camera.get_exposure_time()
            test = camera.get_image()
            max_pixel = np.amax(test)
            fleg += 1
            print('Increasing exposure time, iteration no.: ',fleg, 'New exposure is: ', exposure_time,
                  'Max pixel value :', (int(max_pixel)))
            if int(exposure_time) == max_exp_time:
                print('need to remove nd')
                filter_pos_idx = allowed_filter_positions.index(filter_pos)
                filter_pos = allowed_filter_positions[filter_pos_idx - 1]
                filter_wheel.set_position(filter_pos)
                print('nd pos ' + str(filter_pos))
                camera.set_exposure_time(min(max_exp_time, exposure_time * 1.8))
                exposure_time = camera.get_exposure_time()
                time.sleep(2)
        print('Snap after increasing exposure')
        # image = camera.get_image()
        # timestamp = make_timestamp(precision=0)
        # dataset = ExperimentalDataSet(timestamp)
        # camera_data = CameraData(timestamp,
        #                          extension='_' + str(position) + '_' + str(int(exposure_time)) + '_single' + '.png')
        # camera_data.data = image
        # dataset.dataset["CavityCamera" + str(int(stage.get_position())) + 'single'] = camera_data
        # dataset.saveAllData()
        # print(stage.get_position())
        # real_position.append(stage.get_position())
        # time.sleep(0.2)

filter_wheel.shutdown()
def save_position(array,array_name,folder_path):

    file_path = os.path.join(folder_path,f'{array_name}.txt')
    np.savetxt(file_path,array,fmt='%.4f',delimiter=',')

save_position(real_position,'List_of_position','D:/Data/2024/202408/20240808')

print("End of measurement")




import Components as comp
import socket
import sys

if socket.gethostname() == "ph-photonbec5":
    sys.path.append(r"D:/Control/PythonPackages/")

components = dict()


filter_wheel = comp.FilterWheel(allowed_filter_positions=[0,1,2,3,4,5])


import numpy as np
import Components as comp
import time
import socket
import sys

if socket.gethostname() == "ph-photonbec5":
    sys.path.append("D:/Control/PythonPackages/")
    sys.path.append("D:/Control/PiezoController/")

from pbec_analysis import make_timestamp, ExperimentalDataSet, CameraData


def save_image(image, stage_position, exposure_time):
    timestamp = make_timestamp(precision=0)
    dataset = ExperimentalDataSet(timestamp)
    camera_data = CameraData(timestamp,
                             extension='_' + str(round(stage_position)) + '_' + str(int(exposure_time)) + '_' + str(
                                 np.amax(image)) + '_single' + '.png')
    camera_data.data = image
    dataset.dataset[
        "CavityCamera" + str(stage_position) + str(exposure_time) + str(np.amax(image)) + 'single'] = camera_data
    dataset.saveAllData()
    time.sleep(0.2)


def exposition_check(image_list):  # FOR BOOLEEAN
    overexposed = 0
    good_pics = 0
    underexposed = 0
    bad_pics = 0
    exposition_change = 0
    for k in range(len(image_list)):
        if np.amax(image_list[k]) == 250.0:
            overexposed += 1
            bad_pics += 1
        if np.amax(image_list[k]) < 250.0 and np.amax(image_list[k]) > 120.0:
            good_pics += 1
        if np.amax(image_list[k]) <= 120.0:
            underexposed += 1
            bad_pics += 1
    if overexposed >= round(bad_pics / 2):
        exposition_change = 1  # DECREASE EXPOSURE
    if underexposed >= round(bad_pics / 2):
        exposition_change = 2  # INCREASE EXPOSURE
    result_list = [exposition_change, good_pics, bad_pics, overexposed, underexposed]
    return result_list


position_list = np.arange(221, 90, -1)
number_of_pics = 10
min_exp = 100
max_exp = 1999994
standard_exp = 50000

with comp.Thor_Camera(min_exp, max_exp, measure=True, max_frames=1) as camera:
    with comp.Translation_Stage() as stage:

        for k in range(len(position_list)):
            filter_wheel.reset()
            stage.set(position_list[k])
            camera.change_exposure(min_exp)
            position = stage.get_position()
            exposure = camera.get_exposure()
            iteration = 0
            print('Now at Position : ', position)

            print('Starting verification')
            verification_img_list = camera.get_multiple_images(number_of_pics)

            while exposition_check(verification_img_list)[0] != 0:
                print('Need to readjust parameters, iteration :', iteration)
                verification_img_list = camera.get_multiple_images(number_of_pics)
                verification = exposition_check(verification_img_list)[0]
                if verification == 1:
                    print('need to decrease exposure')
                    camera.change_exposure(exposure * 0.9)
                    exposure = camera.get_exposure()

                    if exposure <= min_exp:
                        print('Minimum exposure reached, increasing filter')
                        filter_wheel.increase_filter()
                        camera.change_exposure(min_exp)

                if verification == 2:
                    print('need to increase exposure')
                    camera.change_exposure(exposure * 1.1)
                    exposure = camera.get_exposure()
                    if exposure >= max_exp:
                        print('Maximum exposure reached, decreasing filter')
                        filter_wheel.decrease_filter()
                        camera.change_exposure(standard_exp)

                iteration += 1

            print('exposition checked for position :', position)

            for k in range(number_of_pics):
                img = camera.get_image()
                save_image(img, position, exposure)
                time.sleep(0.1)
                print('pic num. : ', k, 'for position :', position)


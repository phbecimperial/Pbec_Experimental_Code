import os
import matplotlib.pyplot as plt
import numpy as np
try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path

    configure_path()
except ImportError:
    configure_path = None

import numpy as np

os.add_dll_directory(
    r'C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific Camera Interfaces\SDK\Native Toolkit\dlls\Native_32_lib')
from PIL import Image
import time
import datetime
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK


def get_timestamp(accuracy):
    ts = time.time()
    if accuracy == 'seconds':
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S')
    elif accuracy == 'milliseconds':
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S-%f')
    return st


folder = r'D:\Data\2024\202408'

with TLCameraSDK() as sdk:
    available_cameras = sdk.discover_available_cameras()
    if len(available_cameras) < 1:
        print("no cameras detected")

    with sdk.open_camera(available_cameras[0]) as camera:
        # with sdk.open_camera('17440') as camera:
        camera.exposure_time_us = 100000  # exposure time is in microseconds
        camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
        camera.image_poll_timeout_ms = 4000  # polling timeout # I don't know what this means??
        # old_roi = camera.roi  # store the current roi
        # print(camera.roi)
        print(camera.exposure_time_us)
        """
        uncomment the line below to set a region of interest (ROI) on the camera
        """
        # camera.roi = (324, 324, 454, 454)  # set roi to be at origin point (100, 100) with a width & height of 500

        # if camera.gain_range.max > 0:
        #     db_gain = 0
        #     gain_index = camera.convert_decibels_to_gain(db_gain)
        #     camera.gain = gain_index
        #
        #     #print(f"Set camera gain to {camera.convert_gain_to_decibels(camera.gain)}")
        #
        # camera.arm(2)
        #
        # camera.issue_software_trigger()
        # frames_counted = 0
        camera.arm(2)
        camera.issue_software_trigger()
        frame = camera.get_pending_frame_or_null()
        image_data = frame.image_buffer
        img = Image.fromarray(image_data)
        plt.imshow(img)

        # img.save(folder + "\\"+ 'TEST' + "_" + str(int(round(camera.exposure_time_us/1000,0))) + "ms.png")
        # Put loop to do things here...
        # frame = camera.get_pending_frame_or_null()
        # if frame is None:
        # 	raise TimeoutError("Timeout was reached while polling for a frame, program will now exit")
        # frames_counted += 1
        # image_data = frame.image_buffer
        # img = Image.fromarray(image_data)
        # st = get_timestamp('seconds')
        # img.save(folder + "\\"+ st + "_" + str(int(round(camera.exposure_time_us/1000,0))) + "ms.png")
        # print('Image saved')

        time.sleep(0.1)
        camera.disarm()
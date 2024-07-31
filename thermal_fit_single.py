import numpy as np
import lmfit as lf
from PIL import Image
import glob
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import scipy
from lmfit.lineshapes import gaussian2d
from scipy.ndimage import gaussian_filter
import pickle

cpath = r'D:/Data'
ctstamp = [' 20231213_132429 ', ' 20231213_142448 ']


def opensort(path, timestamps):
    path = path + '/' + timestamps[0][1:5] + '/' + timestamps[0][1:7] + '/' + timestamps[0][1:9] + '/'
    print(path)

    timestamps = [int(i.split('_')[1]) for i in timestamps]
    metas = glob.glob(path + "*meta.json")
    spectra = glob.glob(path + "*spectrum.json")
    pics = glob.glob(path + '*.png')

    metaparams = []
    spectradicts = []
    pic_list = []
    powers = []

    for i, (meta, spectrum, pic) in tqdm(enumerate(zip(metas, spectra, pics))):
        fm = open(meta)
        mdict = json.load(fm)

        if timestamps[0] <= int(mdict['ts'].split('_')[1]) <= timestamps[1]:
            fs = open(spectrum)
            sdict = json.load(fs)

            image = Image.open(pic)
            image_array = np.array(image, dtype=float)
            print(image_array)
            image_array /= float(pic.split('_')[-2])
            pic_list.append(image_array)

            sdict['spectrum'] = np.array(sdict['spectrum'])
            sdict['lamb'] = np.array(sdict['lamb'])
            # sdict['std'] = np.array(sdict['std'])

            metaparams.append(mdict['parameters'])
            spectradicts.append(sdict)

            power = mdict['parameters']['power'] * 1e-3
            powers.append(power)

    sorted_args = np.argsort(powers)
    metaparams, spectradicts, pic_list = (np.array(metaparams)[sorted_args], np.array(spectradicts)[sorted_args],
                                          np.array(pic_list)[sorted_args])

    return metaparams, spectradicts, pic_list


# plt.imshow(pic[8])
# plt.show()


def double_gauss(x, y, amp1, x1, y1, x2, y2, s1, amp2, s2, background):
    add = (s1 * np.sqrt(2 * np.pi)) ** 2 * lf.lineshapes.gaussian2d(x, y, amp1, x1, y1, s1, s1) + \
          (s2 * np.sqrt(2 * np.pi)) ** 2 * lf.lineshapes.gaussian2d(x, y, amp2, x2, y2, s2, s2) + background
    return add


def single_gauss(x, y, amp1, x1, y1, x2, y2, s1, amp2, s2, background):
    add = (s2 * np.sqrt(2 * np.pi)) ** 2 * lf.lineshapes.gaussian2d(x, y, amp2, x2, y2, s2, s2) + background
    return add


def quad_gauss(x, y, amp1, x1, y1, x2, y2, s1, amp2, s2, background):
    add1 = -(2 * amp1 / (s1 ** 2)) * ((x - x1) ** 2 + (y - y1) ** 2) + amp1
    add1[add1 < 0] = 0
    add2 = (s2 * np.sqrt(2 * np.pi)) ** 2 * lf.lineshapes.gaussian2d(x, y, amp2, x2, y2, s2, s2) + background
    return add1 + add2


def fit_gaussian2d(data, fit_func, i=0, sf=0.3, plot=False, crop=100):
    # data = data[0:900, 500:900]

    data[data == 255.0] = 0

    data = data / np.max(data)
    data_spline = data
    data_spline = scipy.ndimage.gaussian_filter(data_spline, 3)
    plt.imshow(data_spline)
    plt.show()
    data_spline = scipy.ndimage.zoom(data_spline, sf)
    plt.imshow(data_spline)
    plt.show()

    # max_args = np.unravel_index(data_spline.argmax(), data_spline.shape)
    # data_spline = data_spline[(max_args[0]-crop):(max_args[0]+crop), (max_args[1]-crop):(max_args[1]+crop)]

    if plot:
        plt.imshow(data_spline)
        plt.title('Data, No fit')
        plt.show()

    if np.max(data_spline) < 1e-5:
        return None
    else:

        y = np.linspace(0, data_spline.shape[1], data_spline.shape[1])
        x = np.linspace(0, data_spline.shape[0], data_spline.shape[0])

        X, Y = np.meshgrid(y, x)

        x_f, y_f, z_f = (X.flatten(), Y.flatten(), data_spline.flatten())

        model1 = lf.Model(fit_func, independent_vars=['x', 'y'])
        model2 = lf.models.Gaussian2dModel()
        params = model2.guess(z_f, y_f, x_f)
        params = params.valuesdict()

        maxarg = np.argmax(z_f)
        print(z_f[maxarg])

        nparams = {'amp1': dict(value=np.max(data_spline) / 2),  # , max=1.3),
                   'y1': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
                   'x1': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
                   'y2': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
                   'x2': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
                   's1': dict(value=5 * sf),  # , min=54 * sf - 1 * sf, max=70 * sf),
                   'amp2': dict(value=np.max(data_spline), vary=False),
                   's2': dict(value=params['sigmax'], min=10 * sf, max=50 * sf),
                   'background': dict(value=0.0, min=0),
                   'Integral': dict(value=np.sum(data_spline), vary=False)
                   }

        # nparams = {'amp1': dict(value=np.max(data_spline) / 2),#, max=1.3),
        #            'y1': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x1': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            'y2': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x2': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            's1': dict(value=5 * sf), #, min=54 * sf - 1 * sf, max=70 * sf),
        #            'amp2': dict(value=0.0, vary=False),
        #            's2': dict(value=params['sigmax'], min=50 * sf, max=800 * sf),
        #            'background': dict(value=0.0, min=0),
        #            'Integral': dict(value=np.sum(data_spline), vary=False)
        #            }

        # nparams = {'amp1': dict(value=np.max(data_spline) / 2, min=0),
        #            'y1': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x1': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            'y2': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x2': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            's1': dict(value=54*sf/2.355, min=(54 * sf - 10 * sf)/2.355, max=(54 * sf + 10 * sf)/2.355),
        #            'amp2': dict(value=np.max(data_spline) / 2, min=0, max=1.3),
        #            's2': dict(value=params['sigmax'], min=100 * sf, max=800 * sf),
        #            'background': dict(value=0.2, min=0, max=0.5),
        #            'Integral': dict(value=np.sum(data_spline), vary=False)
        #            }
        print(nparams)

        nparams = lf.create_params(**nparams)

        result = model1.fit(z_f, x=y_f, y=x_f, params=nparams)

        print(lf.report_fit(result))
        data_peak = np.unravel_index(np.argmax(data_spline), data_spline.shape)

        if plot:
            plt.imshow(model1.func(Y, X, **result.best_values))
            plt.title('Fitted gaussian')
            plt.show()

            plt.plot(model1.func(Y, X, **result.best_values)[int(data_peak[0])], label='fit')
            plt.plot(single_gauss(Y, X, **result.best_values)[int(data_peak[0])], label='thermal cloud')
            plt.plot(data_spline[int(data_peak[0])], label='data')
            plt.legend()
            plt.title(result.values, fontsize=6)
            plt.savefig("Thermal_fits/img{}".format(1000))
            plt.show()
            plt.close()

        else:
            plt.plot(model1.func(Y, X, **result.best_values)[int(data_peak[0])], label='fit')
            plt.plot(single_gauss(Y, X, **result.best_values)[int(data_peak[0])], label='thermal cloud')
            plt.plot(data_spline[int(data_peak[0])], label='data')
            plt.legend()
            plt.title(result.values, fontsize=6)
            plt.savefig("Thermal_fits/img{}".format(1000))
            # plt.show()
            plt.close()

        return result.params


def fit_single(data, i=0, sf=0.3, plot=False, crop_point=[], crop_size=100):
    # data = data[0:900, 500:900]

    data[data == 255.0] = 0

    data = data / np.max(data)
    data_spline = data
    data_spline = scipy.ndimage.gaussian_filter(data_spline, 3)

    data_spline = data_spline[(crop_point[0] - crop_size):(crop_point[0] + crop_size),
                  (crop_point[1] - crop_size):(crop_point[1] + crop_size)]

    plt.imshow(data_spline)
    plt.show()
    data_spline = scipy.ndimage.zoom(data_spline, sf)
    plt.imshow(data_spline)
    plt.show()

    # max_args = np.unravel_index(data_spline.argmax(), data_spline.shape)


    if plot:
        plt.imshow(data_spline)
        plt.title('Data, No fit')
        plt.show()

    if np.max(data_spline) < 1e-5:
        return None
    else:

        y = np.linspace(0, data_spline.shape[1], data_spline.shape[1])
        x = np.linspace(0, data_spline.shape[0], data_spline.shape[0])

        X, Y = np.meshgrid(y, x)

        x_f, y_f, z_f = (X.flatten(), Y.flatten(), data_spline.flatten())

        # model1 = lf.Model(fit_func, independent_vars=['x', 'y'])
        model2 = lf.models.Gaussian2dModel()
        params = model2.guess(z_f, y_f, x_f)
        params.add('scale', value = sf, vary = False)
        params.add('center', value = crop_size, vary = False)
        params.add('scaled_disp', expr = 'sqrt((centerx - center)**2 + (centery-center)**2) *3.45/5 /scale')
        params.add('Scaled_beam_radius', expr = '(sigmax + sigmay) * 3.45/5 / scale')
        result = model2.fit(z_f, x=y_f, y=x_f, params=params)

        # params = params.valuesdict()
        #
        # maxarg = np.argmax(z_f)
        # print(z_f[maxarg])
        #
        # nparams = {'amp1': dict(value=np.max(data_spline) / 2),#, max=1.3),
        #            'y1': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x1': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            'y2': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x2': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            's1': dict(value=5 * sf), #, min=54 * sf - 1 * sf, max=70 * sf),
        #            'amp2': dict(value=np.max(data_spline), vary=False),
        #            's2': dict(value=params['sigmax'], min=10 * sf, max= 50 * sf),
        #            'background': dict(value=0.0, min=0),
        #            'Integral': dict(value=np.sum(data_spline), vary=False)
        #            }

        # nparams = {'amp1': dict(value=np.max(data_spline) / 2),#, max=1.3),
        #            'y1': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x1': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            'y2': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x2': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            's1': dict(value=5 * sf), #, min=54 * sf - 1 * sf, max=70 * sf),
        #            'amp2': dict(value=0.0, vary=False),
        #            's2': dict(value=params['sigmax'], min=50 * sf, max=800 * sf),
        #            'background': dict(value=0.0, min=0),
        #            'Integral': dict(value=np.sum(data_spline), vary=False)
        #            }

        # nparams = {'amp1': dict(value=np.max(data_spline) / 2, min=0),
        #            'y1': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x1': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            'y2': dict(value=x_f[maxarg], min=x_f[maxarg] - 5, max=x_f[maxarg] + 5),
        #            'x2': dict(value=y_f[maxarg], min=y_f[maxarg] - 5, max=y_f[maxarg] + 5),
        #            's1': dict(value=54*sf/2.355, min=(54 * sf - 10 * sf)/2.355, max=(54 * sf + 10 * sf)/2.355),
        #            'amp2': dict(value=np.max(data_spline) / 2, min=0, max=1.3),
        #            's2': dict(value=params['sigmax'], min=100 * sf, max=800 * sf),
        #            'background': dict(value=0.2, min=0, max=0.5),
        #            'Integral': dict(value=np.sum(data_spline), vary=False)
        #            }
        # print(nparams)
        #
        # nparams = lf.create_params(**nparams)

        # result = model1.fit(z_f, x=y_f, y=x_f, params=nparams)

        print(lf.report_fit(result))
        data_peak = np.unravel_index(np.argmax(data_spline), data_spline.shape)

        if plot:
            plt.imshow(model2.func(Y, X, **result.best_values))
            plt.title('Fitted gaussian')
            plt.show()

            plt.plot(model2.func(Y, X, **result.best_values)[int(data_peak[0])], label='fit')
            # plt.plot(single_gauss(Y, X, **result.best_values)[int(data_peak[0])], label='thermal cloud')
            plt.plot(data_spline[int(data_peak[0])], label='data')
            plt.legend()
            plt.title(result.values, fontsize=6)
            # plt.savefig("Thermal_fits/img{}".format(1000))
            plt.show()
            plt.close()

        else:
            plt.plot(model2.func(Y, X, **result.best_values)[int(data_peak[0])], label='fit')
            # plt.plot(single_gauss(Y, X, **result.best_values)[int(data_peak[0])], label='thermal cloud')
            plt.plot(data_spline[int(data_peak[0])], label='data')
            plt.legend()
            plt.title(result.values, fontsize=6)
            # plt.savefig("Thermal_fits/img{}".format(1000))
            # plt.show()
            plt.close()

        return result.params


# Insert pickling here
# ldpickle = False

# No convolution!

a = 0
outputs = []
# if not ldpickle:
# mps, sps, pic = opensort(cpath, ctstamp)
# pic = pic[6]

# pic = np.array(Image.open(r'D:\Data\2024\202402\Pump_spot_displaced_20240222.bmp'), dtype=float)

aligned_pic = np.array(Image.open(r'C:\Data\2024\PumpSpotLocation\30_07_aligned.bmp'), dtype=float)

pic = np.array(Image.open(r'C:\Data\2024\PumpSpotLocation\30_07_aligned.bmp'), dtype=float)

plt.imshow(pic)
plt.title('Data')
plt.show()

crop_point = np.array(np.unravel_index(aligned_pic.argmax(), aligned_pic.shape))
crop_point[0] -= 0 #Fix displacement
crop_point[1] += 17
print(crop_point)

result = fit_single(pic, i=0, sf=1, plot=True, crop_point=crop_point, crop_size=200)  # What you change

print('result')

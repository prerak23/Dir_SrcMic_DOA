##########################################################################################
# Calculate GCC-PHAT results by modifying a specific GCC-PHAT function of pyroomacoustics#
##########################################################################################


import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import h5py
from scipy.signal import stft
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
import seaborn as sns

sns.set_theme()
import math
from scipy.spatial import distance
from scipy import signal


def polar2cart(phi, theta, r):
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)

    return [
        r * math.cos(theta) * math.cos(phi),
        r * math.cos(theta) * math.sin(phi),
        r * math.sin(theta),
    ]


number_of_sources = 1
fs = 16000
nfft = 1536
tolerance = 5
d = 0.104
phi = 0
M = 2
pred_gt = []
c = 343.0
freq_bins = np.arange(30, 1024)


pred_gt = []
# vh_mic=np.array([voicehome2_mic_1_,voicehome2_mic_2_]).T


acc = 0
err = 0
co = 0

##########################
# Retreive DIRHA data    #
##########################

path = (
    "/home/psrivastava/source_localization/doa_estimation/DIRHA_mic_arr/test_scripts/"
)
data = np.load(path + "data_DIRHA.npy")
coord = np.load(path + "DOA_DIRHA.npy")
record_ = np.load(path + "record_data.npy")
all_doa = []
all_tdoa = []
number_of_samples = round(48000 * 16 / 24)

speed_of_sound = 343

distance_between_mic = 0.30  # V.imp distance between mics in the array.

for b in np.arange(data.shape[0]):

    data_16k = data[b, :, :]

    azimuth_src = coord[b, 0]

    # Modified function of pyroomacoustics

    delay = pra.experimental.localization.tdoa(
        data_16k[0, :], data_16k[1, :], distance_between_mic, interp=4, fs=16000
    )

    # if delay > -0.0008:
    all_tdoa.append([azimuth_src, delay])

    # Calculate estimated doa using the estimated delay, calculated using GCC-PHAT (Modified function of pyroomacoustics)

    estimated_angle = np.rad2deg(
        np.arccos((delay * speed_of_sound) / (distance_between_mic))
    )

    # print(azimuth_src,estimated_angle)

    if not np.isnan(estimated_angle):
        angular_err = np.abs(azimuth_src - estimated_angle)

        err += angular_err

        if angular_err <= 20:
            acc += 1
        elif angular_err > 80:
            # wavfile.write("Check_cluster.wav",16000,data_16k[1,:].astype(np.int16))
            print(record_[b, :], angular_err, estimated_angle, azimuth_src)

        all_doa.append([azimuth_src, estimated_angle])

        # print(delay)
        # wavfile.write("-2_delay.wav",24000,data_16k[0,:])
        # break

        # stft_2_mic=np.zeros((2,769,33))
        # stft_2_mic[0,:,:]= stft(data_16k[0,:24000],fs=24000,nperseg=1536)[2]
        # stft_2_mic[1,:,:]= stft(data_16k[1,:24000],fs=24000,nperseg=1536)[2]

        # err,acc=srp_phat(err,acc,stft_2_mic,[azimuth_src])

        co += 1


#################################################
# Calculate confidence interval and other stats #
#################################################

import math

all_tdoa = np.array(all_tdoa)
all_doa = np.array(all_doa)
plt.scatter(all_tdoa[:, 0], all_tdoa[:, 1])
# plt.hist(all_doa,bins=np.linspace(0,180,num=30),color="blue")
errs = np.abs(all_doa[:, 0] - all_doa[:, 1])
mean = np.mean(errs)
std = np.std(errs)

ci_95 = 1.96 * std / math.sqrt(all_doa.shape[0])
print("CI", ci_95)

plt.savefig("all_tdoa_DIRHA_decimate.jpeg")

plt.clf()
plt.scatter(all_doa[:, 0], all_doa[:, 1])
plt.xlabel("Ground Truth DOA")
plt.ylabel("Estimated DOA")
plt.plot(np.linspace(0, 180, num=360), np.linspace(0, 180, num=360), color="orange")
plt.savefig("err_decimate.jpeg")

print("Accuracy over validation set SRP-PHAT", (acc / co) * 100)
print("Mean angular error over validation set SRP-PHAT", (err / co))

#####################################################################################################
# Calculate performance of the trained network over the generated simulated validation/test set     #
#####################################################################################################


import numpy as np
import torch
import h5py
import he_cnn_resnet as net_

import torch.nn as nn
import torch.optim as optim

from scipy.io import wavfile
from scipy import signal
import math
from scipy.spatial import distance


device = torch.device("cuda")
net = net_.he_archs(input_size=(4, 33, 1025)).to(torch.device(device))

path_to_chkp = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/voicehome2_arr/training/D6_0111/train_again/he_cnn_save_best_sh_106.pt"

chkp = torch.load(path_to_chkp, map_location=device)

net.load_state_dict(chkp["model_dict"])
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
optimizer.load_state_dict(chkp["optimizer_dic"])

net.eval()

window_len = 2048  # Equals to 128ms in 1 second of signal sampled @ 16khz
time_segment_stft = (
    33  # For 1 second of signal and 50 % overlap with window_len of 2048
)
max_channel_data = 2
device = "cpu"
ACT_NONE = 0
_STAGE1_LOSS = nn.MSELoss()
batch_size = 64
no_of_epochs = 110
sigmoid = nn.Sigmoid()

#############################################
#     Calculate features for DNNs           #
#############################################


def cal_features(data):

    data_stft_real = np.zeros(
        (data.shape[0], max_channel_data, window_len // 2 + 1, time_segment_stft)
    )
    data_stft_imag = np.zeros(
        (data.shape[0], max_channel_data, window_len // 2 + 1, time_segment_stft)
    )

    for batch in np.arange(data.shape[0]):
        data_stft = signal.stft(
            data[batch, :, :], axis=-1, fs=16000, nperseg=window_len
        )

        data_stft_real[batch, :, :, :] = np.real(data_stft[2])
        data_stft_imag[batch, :, :, :] = np.imag(data_stft[2])

    x = np.concatenate((data_stft_real, data_stft_imag), axis=1)
    x = np.transpose(
        x, (0, 1, 3, 2)
    )  # Dimension (Batch_size x 4 {2 real and 2 imag coeffs} x time_frames x freq_bins )

    return torch.tensor(x).to(device=device)


######################################
# Spherical to cartesian coordinate  #
#  V.carefull with this stuff        #
######################################


def polar2cart(phi, theta, r):
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)

    return [
        r * math.cos(theta) * math.cos(phi),
        r * math.cos(theta) * math.sin(phi),
        r * math.sin(theta),
    ]


######################################
# Load simulated/validation test set #
######################################

path = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/"
dataset = "D3_1011"
data = h5py.File(
    path + dataset + "/noisy_mixtures/" + dataset + "_aggregated_mixture.hdf5"
)
coord = h5py.File(
    path
    + dataset
    + "/noisy_mixtures/"
    + dataset
    + "_aggregated_mixture_annotations.hdf5"
)


#########################################################
# Load the file that consist of which rooms to test for #
#########################################################

rooms = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/data_generation/val_random_ar_only_source_1_included_new.npy"
)

data_16k_arr = np.zeros((rooms.shape[0], 2, 32000))
doa_16k = []


for j, b in enumerate(rooms):
    b = b[0]
    one_example_from_one_room = data["room_nos"][b]["nsmix_f"][0, :, :]

    # Retreive ground truth doa
    azimuth_src = coord["room_nos"][b]["azimuth"][0]

    data_16k_arr[j, :, :] = one_example_from_one_room
    doa_16k.append([azimuth_src])


doa_16k = np.array(doa_16k)
import matplotlib.pyplot as plt

stft_feat = cal_features(data_16k_arr)

output_spectrum_stack = np.zeros((stft_feat.shape[0], 360))

for j in np.arange(stft_feat.shape[0]):

    data_unsq = torch.unsqueeze(
        stft_feat[j, :, :, :], axis=0
    )  # Add one extra dimension at axis=0, so that it behaves like batch of just one example, when input to the network
    # print(data_unsq.shape,j)
    output_spectrum = net(
        data_unsq.cuda().float()
    )  # Procees one example at a time, due to memory constraints of the GPU
    output_spectrum_stack[j, :] = output_spectrum.detach().cpu().numpy()


# Use the same metric as done for other tests.
acc, err, c, d = net_.doa_metric(output_spectrum_stack, doa_16k, "test")
c = np.array(c)

errs = np.abs(c[:, 0] - c[:, 1])
mean = np.mean(errs)
std = np.std(errs)

ci_95 = 1.96 * std / math.sqrt(doa_16k.shape[0])

plt.scatter(c[:, 1], c[:, 0])

plt.plot(np.linspace(0, 180, num=360), np.linspace(0, 180, num=360), color="orange")
plt.ylabel("Predicted")
plt.xlabel("GT DoA")
plt.savefig("scatter_vc.test_dcase.jpeg")
print("Accuracy ", acc)
print("Angular Error ", err)
print("Confidence Interval 95", mean, ci_95)

print(d)

################################################################################
# Calculate performance of the trained network over the test set of voiceHome2 #
################################################################################

import numpy as np
import torch
import h5py

# import data_loader_test
import he_cnn_resnet as net_

import torch.nn as nn
import torch.optim as optim

from scipy.io import wavfile
from scipy import signal


device = torch.device("cpu")
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

    # Shape data : (no_samples, no_channels, 32000) For 2 second signals

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


#####################################
# Load voicehome data                #
#####################################

audio_path = "/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/voiceHome2/orig_corpora/audio/noisy/"
annotation_path = "/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/voiceHome2/orig_corpora/annotations/rooms/"

fs = 16000
sLen = 16000
no_of_sources = 1
err = 0
acc = 0


nHouse = 4  # number of houses

nRoom = 3  # number of rooms per house

nSpk = 3  # number of speakers per house

nPos = 5  # number of speakers positions

nNoise = 4  # number of noise conditions per room

nUtt = 2  # number of utterances per {spk,pos,room,house,noise}

err = 0
acc = 0

enc = [
    ["F1", "M1", "M2"],
    ["F2", "M3", "M4"],
    ["F3", "M5", "M6"],
    ["F4", "M7", "M8"],
]  # Encoding of Female and Male speakers in 4 different houses.


voicehome_home1 = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/test_scripts/real_data/voiceHome2_house_1.npy",
    allow_pickle=True,
)
voicehome_home1_ano = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/test_scripts/real_data/voiceHome2_house_1_ano.npy",
    allow_pickle=True,
)

voicehome_home2 = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/test_scripts/real_data/voiceHome2_house_2.npy",
    allow_pickle=True,
)
voicehome_home2_ano = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/test_scripts/real_data/voiceHome2_house_2_ano.npy",
    allow_pickle=True,
)

voicehome_home3 = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/test_scripts/real_data/voiceHome2_house_3.npy",
    allow_pickle=True,
)
voicehome_home3_ano = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/test_scripts/real_data/voiceHome2_house_3_ano.npy",
    allow_pickle=True,
)

voicehome_home4 = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/test_scripts/real_data/voiceHome2_house_4.npy",
    allow_pickle=True,
)
voicehome_home4_ano = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/test_scripts/real_data/voiceHome2_house_4_ano.npy",
    allow_pickle=True,
)


vh_arr = [
    voicehome_home1,
    voicehome_home2,
    voicehome_home3,
    voicehome_home4,
]  # List of variables
vh_ano_arr = [
    voicehome_home1_ano,
    voicehome_home2_ano,
    voicehome_home3_ano,
    voicehome_home4_ano,
]


co = 0
l = 0
real_data_stack = np.empty((1, 2, 32000))
real_data_anno_stack = np.empty((1, 1))

for h_no in np.arange(nHouse):  # Iterate through homes

    # Pick particular home array and it's annotation

    vh = vh_arr[h_no]
    vh_ano = vh_ano_arr[h_no]

    for r_no in np.arange(nRoom):
        # Iterate through rooms, each home has 3 rooms.

        r_enc = r_no + 1

        for spk_no in np.arange(nSpk):
            # Iterate through number of speakers, each home has 3 spakers

            spk_enc = enc[h_no][spk_no]

            for pos_no in np.arange(nPos):

                # Iterate thorugh each speaker positions,each speaker has 5 position.
                # Below we get all the annotations.

                # Retreive gt angle
                doa_angle_gt = [
                    [
                        vh_ano.item()
                        .get("room_" + str(r_enc))
                        .get(spk_enc)
                        .get("position_" + str(pos_no + 1))
                        .get("doa_angle")
                    ]
                ]

                # Retreive mic positions
                mics = np.array(
                    [
                        vh_ano.item()
                        .get("room_" + str(r_enc))
                        .get(spk_enc)
                        .get("position_" + str(pos_no + 1))
                        .get("mic_1"),
                        vh_ano.item()
                        .get("room_" + str(r_enc))
                        .get(spk_enc)
                        .get("position_" + str(pos_no + 1))
                        .get("mic_2"),
                    ]
                ).T

                # Retreive barycenter position
                bc = (
                    vh_ano.item()
                    .get("room_" + str(r_enc))
                    .get(spk_enc)
                    .get("position_" + str(pos_no + 1))
                    .get("array_cord_bc")
                )

                # Retreive speaker coordinate
                spk_dist = (
                    vh_ano.item()
                    .get("room_" + str(r_enc))
                    .get(spk_enc)
                    .get("position_" + str(pos_no + 1))
                    .get("spk_cord")
                )

                # Retreive specific .wav file
                for j in (
                    vh.item()
                    .get("room_" + str(r_enc))
                    .get(spk_enc)
                    .get("position_" + str(pos_no + 1))
                ):

                    if (
                        "noiseCond1_" in j
                    ):  # If noiseCond1_ is satisfied : means it is for quite noise.
                        wav_file_path = audio_path + j

                        samplerate, data = wavfile.read(wav_file_path, "r")
                        data_2_mic = np.expand_dims(data[80000:112000, :2].T, axis=0)

                        real_data_stack = np.vstack((real_data_stack, data_2_mic))

                        real_data_anno_stack = np.vstack(
                            (real_data_anno_stack, doa_angle_gt)
                        )


import matplotlib.pyplot as plt
import math

# Calculate features
stft_feat = cal_features(real_data_stack[1:, :, :])
# Output spectrum from the network
output_spectrum = net(stft_feat.float())

# Metric outputs recall, mae and c: array of (estimated_doa, gt_doa)

acc, err, c, d = net_.doa_metric(output_spectrum, real_data_anno_stack[1:, :], "test")
c = np.array(c)

# Calculate 95 Confidence interval over array of error.
errs = np.abs(c[:, 0] - c[:, 1])
mean = np.mean(errs)
std = np.std(errs)

ci_95 = 1.96 * std / math.sqrt(real_data_stack.shape[0] - 1)


plt.scatter(c[:, 1], c[:, 0])

plt.plot(np.linspace(0, 180, num=360), np.linspace(0, 180, num=360), color="orange")
plt.ylabel("Predicted")
plt.xlabel("GT DoA")
plt.savefig("scatter_vc.test.jpeg")
print("Accuracy ", acc)
print("Angular Error ", err)

print("Confidence Interval 95", mean, ci_95)

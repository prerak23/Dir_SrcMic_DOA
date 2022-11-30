import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import h5py
from scipy.signal import stft
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
import seaborn as sns

sns.set_theme()

# For simulated dataset
# dataset="D1_0000"
# path_mixture="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/voicehome2_arr/"+dataset+"/noisy_mixtures/"+dataset+"_aggregated_mixture.hdf5"
# path_anotation="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/voicehome2_arr/"+dataset+"/noisy_mixtures/"+dataset+"_aggregated_mixture_annotations.hdf5"

audio_path = "/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/voiceHome2/orig_corpora/audio/noisy/"
annotation_path = "/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/voiceHome2/orig_corpora/annotations/rooms/"


number_of_sources = 1
fs = 16000
nfft = 2048
tolerance = 20
d = 0.104
phi = 0
M = 2
pred_gt = []
c = 343.0
freq_bins = np.arange(30, 1024)
voicehome2_mic_1_ = np.array([0.037, 0.056, -0.038])
voicehome2_mic_2_ = np.array([-0.034, 0.056, 0.038])
vh_mic = np.array([voicehome2_mic_1_, voicehome2_mic_2_]).T

############################
# SRP-PHAT pyroomacoustics #
############################


def srp_phat(err, acc, rec_pos, bc, azimuth_estm, X, j, l, spk_dist):

    # Create a 2D linear array w.r.t to the barycenter and the spacing between the mics.

    L = pra.linear_2D_array(bc[:2], 2, 0, 0.104)

    # print("barycenter",euclidean(rec_pos[:,0],rec_pos[:,1]))
    # print("L",euclidean(L[:,0],L[:,1]))
    # print(rec_pos[:2,:])
    # print(X.shape)

    doa = pra.doa.algorithms.get("SRP")(
        L,
        fs,
        nfft,
        num_src=number_of_sources,
        dim=2,
        mode="far",
        azimuth=np.deg2rad(np.linspace(0, 180, num=360)),
    )

    doa.locate_sources(X)  # ,freq_bins=freq_bins)

    estimated_angle = np.sort(doa.azimuth_recon) / np.pi * 180

    # if estimated_angle[0] > 180:
    #    estimated_angle=360-estimated_angle

    # estimated_angle=180-estimated_angle

    # print('Speakers at: ',estimated_angle, 'degrees')
    # print('Gt_doa',azimuth_estm)

    # estimated_angle=np.abs(180-estimated_angle)
    pred_gt.append([estimated_angle[0], azimuth_estm[0]])

    angular_diff = np.abs(estimated_angle[0] - azimuth_estm[0])
    if angular_diff <= tolerance:
        acc += 1
        err = err + angular_diff
    else:
        # print(j)
        err = err + angular_diff
        l = l + 1

    return acc, err


# mixture_file=h5py.File(path_mixture,"r")
# annotation_file=h5py.File(path_anotation,"r")
# validation_rooms=np.load("/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/data_generation/val_random_ar_only_source_1_included.npy")

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

enc = [["F1", "M1", "M2"], ["F2", "M3", "M4"], ["F3", "M5", "M6"], ["F4", "M7", "M8"]]

# Load voicehome2 datasets from numpy arrays. (Calculated in the file named "get_voice_home_data.py")

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

vh_arr = [voicehome_home1, voicehome_home2, voicehome_home3, voicehome_home4]
vh_ano_arr = [
    voicehome_home1_ano,
    voicehome_home2_ano,
    voicehome_home3_ano,
    voicehome_home4_ano,
]
co = 0
l = 0
for h_no in np.arange(nHouse):
    vh = vh_arr[h_no]
    vh_ano = vh_ano_arr[h_no]

    for r_no in np.arange(nRoom):
        r_enc = r_no + 1

        for spk_no in np.arange(nSpk):
            spk_enc = enc[h_no][spk_no]

            for pos_no in np.arange(nPos):
                doa_angle_gt = [
                    vh_ano.item()
                    .get("room_" + str(r_enc))
                    .get(spk_enc)
                    .get("position_" + str(pos_no + 1))
                    .get("doa_angle")
                ]
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
                bc = (
                    vh_ano.item()
                    .get("room_" + str(r_enc))
                    .get(spk_enc)
                    .get("position_" + str(pos_no + 1))
                    .get("array_cord_bc")
                )
                spk_dist = (
                    vh_ano.item()
                    .get("room_" + str(r_enc))
                    .get(spk_enc)
                    .get("position_" + str(pos_no + 1))
                    .get("spk_cord")
                )

                for j in (
                    vh.item()
                    .get("room_" + str(r_enc))
                    .get(spk_enc)
                    .get("position_" + str(pos_no + 1))
                ):
                    if "noiseCond1_" in j:
                        wav_file_path = audio_path + j

                        samplerate, data = wavfile.read(wav_file_path)
                        data_2_mic = data[96000:112000, :2].T

                        stft_2_mic = np.zeros((2, 1025, 17), dtype=np.complex_)
                        stft_2_mic[0, :, :] = stft(
                            data_2_mic[0, :], fs=16000, nperseg=2048
                        )[2]
                        stft_2_mic[1, :, :] = stft(
                            data_2_mic[1, :], fs=16000, nperseg=2048
                        )[2]

                        acc, err = srp_phat(
                            err, acc, mics, bc, doa_angle_gt, stft_2_mic, j, l, spk_dist
                        )

                        co += 1

print(pred_gt)
pred_gt = np.array(pred_gt)
print(pred_gt.shape)
print("Accuracy over validation set SRP-PHAT", (acc / co) * 100)
print("Mean angular error over validation set SRP-PHAT", (err / co))
print(co)
import math

errs = np.abs(pred_gt[:, 0] - pred_gt[:, 1])
mean = np.mean(errs)
std = np.std(errs)

ci_95 = 1.96 * std / math.sqrt(pred_gt.shape[0])

"""
plt.scatter(pred_gt[:,1],pred_gt[:,0])
plt.plot(np.linspace(0,180,num=360),np.linspace(0,180,num=360),c="red")
plt.xlabel("GT DoA")
plt.ylabel("Estm DoA")

plt.savefig("abc.jpeg")
"""
print(l)
print(ci_95)


"""

for j in np.arange(validation_rooms.shape[0]):
    s = np.zeros((2,1025,17),dtype=np.float32)
    #Commented out since test1 and 2 are only 1minute long
    s[0,:,:] = stft(mixture_file["room_nos"]["room_"+str(j)]["nsmix_f"][no_of_sources-1,0,:],fs=16000,nperseg=2048)[2] #[0*sLen:1*sLen]
    s[1,:,:] = stft(mixture_file["room_nos"]["room_"+str(j)]["nsmix_f"][no_of_sources-1,1,:],fs=16000,nperseg=2048)[2] #[0*sLen:1*sLen]

#Where is the relative center inside the room (around which microphones are located)?
#We know from schematic (https://www.researchgate.net/figure/The-layout-of-the-UEDIN-Instrumented-Meeting-Room-measurements-in-cm-Array_fig1_4208275)
#That the room measurements are 650cm X and 490cm Y

    rec_pos = annotation_file["room_nos"]["room_"+str(j)]["rec_pos"][()].T

    azimuth_estm = [annotation_file["room_nos"]["room_"+str(j)]["azimuth"][:no_of_sources][()]] #A subjective guess where the speaker could be
    room_dimension = annotation_file["room_nos"]["room_"+str(j)]["room_dimension"][()]/2

    center= room_dimension[:2]  #annotation_file["room_nos"]["room_"+str(j)]["barycenter"][()][0,:2]
    d = 0.104     #distance between microphones (in meters)
    mode = 'linear' #define microphone layout (circular or linear)
    nfft = 2048 #size of fft used

    #Test scenario , give array position directily 3 and 2 try both, after that give linear array as a parameters with provided center of the room.

    acc,err=srp_phat(err,acc,rec_pos,center,azimuth_estm,s)



print("Accuracy over validation set SRP-PHAT",(acc/j)*100)
print("Mean angular error over validation set SRP-PHAT",(err/j))
#return detected_windows
np.save("pred_gt.npy",np.array(pred_gt))
"""

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def angular_spectrum(angles_3_src, switch=None):

    if "azimuth" in switch:
        azi_grid = np.linspace(0, 180, num=360)
        azi_angular_spectrum = np.empty((3, 360))
        sigma = 5  # Controls the width of the gaussian curve, the value is in degrees

        angle_distance_1 = np.abs(azi_grid - angles_3_src[0])
        angle_distance_2 = np.abs(azi_grid - angles_3_src[1])
        angle_distance_3 = np.abs(azi_grid - angles_3_src[2])

        azi_angular_spectrum[0, :] = np.exp(-(angle_distance_1 ** 2 / sigma ** 2))
        azi_angular_spectrum[1, :] = np.max(
            (
                np.exp(-(angle_distance_1 ** 2 / sigma ** 2)),
                np.exp(-(angle_distance_2 ** 2 / sigma ** 2)),
            ),
            axis=0,
        )
        azi_angular_spectrum[2, :] = np.max(
            (
                np.exp(-(angle_distance_1 ** 2 / sigma ** 2)),
                np.exp(-(angle_distance_2 ** 2 / sigma ** 2)),
                np.exp(-(angle_distance_3 ** 2 / sigma ** 2)),
            ),
            axis=0,
        )

        return azi_angular_spectrum

    elif "elevation" in switch:
        ele_grid = np.linspace(0, 90, num=360)
        ele_angular_spectrum = np.empty((3, 360))
        sigma_ele = 2
        angles_3_src_ele_new = [180 - n if n > 90 else n for n in angles_3_src]

        angle_distance_1 = np.abs(ele_grid - angles_3_src_ele_new[0])
        angle_distance_2 = np.abs(ele_grid - angles_3_src_ele_new[1])
        angle_distance_3 = np.abs(ele_grid - angles_3_src_ele_new[2])

        ele_angular_spectrum[0, :] = np.exp(-(angle_distance_1 ** 2 / sigma_ele ** 2))
        ele_angular_spectrum[1, :] = np.max(
            (
                np.exp(-(angle_distance_1 ** 2 / sigma_ele ** 2)),
                np.exp(-(angle_distance_2 ** 2 / sigma_ele ** 2)),
            ),
            axis=0,
        )
        ele_angular_spectrum[2, :] = np.max(
            (
                np.exp(-(angle_distance_1 ** 2 / sigma_ele ** 2)),
                np.exp(-(angle_distance_2 ** 2 / sigma_ele ** 2)),
                np.exp(-(angle_distance_3 ** 2 / sigma_ele ** 2)),
            ),
            axis=0,
        )

        return ele_angular_spectrum


# -----------Change this path to aggregate different batch datasets -------------------------------

path_noisy_mix = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/D6_0111/noisy_mixtures/"
new_file = h5py.File(path_noisy_mix + "D6_0111_aggregated_mixture.hdf5", "w")
room_save = new_file.create_group("room_nos")

# snr_src_1_arr_2_mic=[]
# snr_src_2_arr_2_mic=[]
# snr_src_3_arr_2_mic=[]

snr_src_1_mix = []

# SINR
snr_src_1_arr_3_mic = []
snr_src_2_arr_3_mic = []
snr_src_3_arr_3_mic = []
sinr_avg = []


parallel_jobs = 100
nj = 0
for file_name in os.listdir(path_noisy_mix):
    if "noisy" in file_name:
        abc = h5py.File(path_noisy_mix + file_name, "r")
        room_start = int(file_name.split("_")[3])
        print(file_name, nj)
        for i in range(
            room_start, room_start + parallel_jobs
        ):  # File name used to iterate through rooms

            room_mixture = abc["room_nos"]["room_" + str(i)]["nsmix_f"][()]
            room_id = room_save.create_group("room_" + str(i))

            if room_mixture.shape[1] == 2:
                room_id.create_dataset("nsmix_f", (3, 2, 32000), data=room_mixture)
                snr_src_1_mix.append(
                    abc["room_nos"]["room_" + str(i)]["nsmix_snr_src_1"][()]
                )
                snr_src_1_arr_3_mic.append(
                    abc["room_nos"]["room_" + str(i)]["SINR_S1"][()]
                )
                snr_src_2_arr_3_mic.append(
                    abc["room_nos"]["room_" + str(i)]["SINR_S2"][()]
                )
                snr_src_3_arr_3_mic.append(
                    abc["room_nos"]["room_" + str(i)]["SINR_S3"][()]
                )
                sinr_avg.append(abc["room_nos"]["room_" + str(i)]["SINR_AVG"][()])

        nj += 1

"""
else:
    room_id.create_dataset("nsmix_f",(3,2,16000),data=room_mixture)
    snr_src_1_arr_2_mic.append(abc["room_nos"]["room_"+str(i)]["nsmix_snr_src_1"][()])
    snr_src_2_arr_2_mic.append(abc["room_nos"]["room_"+str(i)]["nsmix_snr_src_2"][()])
    snr_src_3_arr_2_mic.append(abc["room_nos"]["room_"+str(i)]["nsmix_snr_src_3"][()])
"""

# Aggregate annotations"
# -----------Change this path to aggregate different batch datasets -------------------------------
path_noisy_mix = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/D6_0111/"
new_file_anno = h5py.File(
    path_noisy_mix + "/noisy_mixtures/" + "D6_0111_aggregated_mixture_annotations.hdf5",
    "w",
)
room_save = new_file_anno.create_group("room_nos")
parallel_jobs = 100
vol_arr = np.empty((1, 40000))
surf_area = np.empty((1, 40000))
rt60 = np.empty((6, 40000))

count_room_no = 0
for file_name in os.listdir(path_noisy_mix):
    if "_anno_rooms_" in file_name:
        abc = h5py.File(path_noisy_mix + file_name, "r")

        for room_no in abc[
            "rirs_save_anno"
        ].keys():  # Keys used to iterate through rooms.
            sa = abc["rirs_save_anno"][room_no]["surf_area"][0]
            vol = abc["rirs_save_anno"][room_no]["volume"][0]
            rt60_ = abc["rirs_save_anno"][room_no]["rt60_median"][()]
            azimuth_3_sources = abc["rirs_save_anno"][room_no]["azimuth"][()]
            barycenter = abc["rirs_save_anno"][room_no]["barycenter"][()]
            rec_pos = abc["rirs_save_anno"][room_no]["rec_pos"][()]
            r_d = abc["rirs_save_anno"][room_no]["room_dimension"][()]

            azimuth_spectrum_3_sources = angular_spectrum(
                azimuth_3_sources, switch="azimuth"
            )

            surf_area[0, count_room_no] = sa
            vol_arr[0, count_room_no] = vol
            rt60[0, count_room_no] = rt60_[0, 0]
            rt60[1, count_room_no] = rt60_[0, 1]
            rt60[2, count_room_no] = rt60_[0, 2]
            rt60[3, count_room_no] = rt60_[0, 3]
            rt60[4, count_room_no] = rt60_[0, 4]
            rt60[5, count_room_no] = rt60_[0, 5]

            room_id = room_save.create_group(room_no)  # Important step
            room_id.create_dataset("surface", 1, data=sa)
            room_id.create_dataset("volume", 1, data=vol)
            room_id.create_dataset("rt60", (1, 6), data=rt60_)
            room_id.create_dataset(
                "azimuth_spectrum", (3, 360), data=azimuth_spectrum_3_sources
            )
            room_id.create_dataset("azimuth", 3, data=azimuth_3_sources)
            room_id.create_dataset("barycenter", (1, 3), data=barycenter)
            room_id.create_dataset("rec_pos", (2, 3), data=rec_pos)
            room_id.create_dataset("room_dimension", 3, data=r_d)

            count_room_no += 1


print("Volume", np.std(vol_arr), np.var(vol_arr))
print("Surface", np.std(surf_area), np.var(surf_area))
print("RT 60 125", np.std(rt60[0, :]), np.var(rt60[0, :]))
print("RT 60 250", np.std(rt60[1, :]), np.var(rt60[1, :]))
print("RT 60 500", np.std(rt60[2, :]), np.var(rt60[2, :]))
print("RT 60 1000", np.std(rt60[3, :]), np.var(rt60[3, :]))
print("RT 60 2000", np.std(rt60[4, :]), np.var(rt60[4, :]))
print("RT 60 4000", np.std(rt60[5, :]), np.var(rt60[5, :]))

# new_snr_src_1_arr_3_mic=[10*np.log10(10**(x/10)-1) for x in snr_src_1_arr_3_mic]

############################################
# Change this path before running the script#
############################################
path_save_figs = "/home/psrivastava/source_localization/doa_estimation/dcase_mic_arr/data_generation/D6_0111/"

"""
fig,ax2=plt.subplots()
ax2.hist(np.array(new_snr_src_1_arr_3_mic),bins=np.arange(100,step=5),color="orange")
ax2.set_ylabel("No of samples")
ax2.set_xlabel("Negative log of variance -log(np.var(sp_signal))")
fig.tight_layout()
plt.savefig("hist_snr_src_1_ARR_3_mic_recal.jpeg")
"""

plt.clf()
fig, ax2 = plt.subplots()
ax2.hist(
    np.array(snr_src_1_arr_3_mic), bins=np.arange(-50, 100, step=5), color="orange"
)
ax2.set_ylabel("No of samples")
ax2.set_xlabel("SINR dB SRC 1")
fig.tight_layout()
plt.savefig(path_save_figs + "SINR_S1.jpeg")

plt.clf()
fig, ax3 = plt.subplots()
ax3.hist(
    np.array(snr_src_2_arr_3_mic), bins=np.arange(-50, 100, step=5), color="orange"
)
ax3.set_xlabel("SINR dB SRC 2")
plt.savefig(path_save_figs + "SINR_S2.jpeg")

plt.clf()
fig, ax4 = plt.subplots()
ax4.hist(
    np.array(snr_src_3_arr_3_mic), bins=np.arange(-50, 100, step=5), color="orange"
)
ax4.set_xlabel("SINR dB SRC 3")
plt.savefig(path_save_figs + "SINR_S3.jpeg")

plt.clf()
# Average histogram
concat_histograms = np.concatenate(
    (
        np.array(snr_src_1_arr_3_mic),
        np.array(snr_src_2_arr_3_mic),
        np.array(snr_src_3_arr_3_mic),
    ),
    axis=0,
)
hist, bins = np.histogram(concat_histograms)
hist_normalized = hist / 3

width = 0.9 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
fig, ax5 = plt.subplots()
ax5.bar(center, hist_normalized, align="center", width=width)
plt.savefig(path_save_figs + "SINR_AVG_normalized_concate_histogram.jpeg")

plt.clf()
fig, ax6 = plt.subplots()
ax6.bar(center, hist, align="center", width=width)
plt.savefig(path_save_figs + "SINR_AVG_concate_histogram.jpeg")


plt.clf()
fig, ax7 = plt.subplots()
ax7.hist(np.array(sinr_avg), bins=np.arange(-30, 40, step=0.5), color="orange")
ax7.set_xlabel("SINR dB AVG SRC 1 2 3")
plt.savefig(path_save_figs + "SINR_AVG.jpeg")

plt.clf()
fig, ax8 = plt.subplots()
ax8.hist(np.mean(rt60, axis=0), bins=np.arange(0, 2.5, step=0.04), color="orange")
ax8.set_xlabel("RT60_in_seconds")
plt.savefig(path_save_figs + "RT60_avg.jpeg")

plt.clf()
fig, ax9 = plt.subplots()
ax9.hist(vol_arr[0, :], bins=np.arange(0, 500, step=5), color="orange")
ax9.set_xlabel("Volume_in_m3")
plt.savefig(path_save_figs + "Volume_histogram.jpeg")

plt.clf()
fig, ax10 = plt.subplots()
ax10.hist(concat_histograms, bins=np.arange(-30, 45, step=2), color="orange")
ax10.set_xlabel("Volume_in_m3")
plt.savefig(path_save_figs + "SINR_AVG_concat_SINR_1_2_3.jpeg")

plt.clf()
fig, ax11 = plt.subplots()
ax11.hist(np.array(snr_src_1_mix), bins=np.arange(-20, 90, step=2), color="orange")
ax11.set_xlabel("SNR_src_1_mix")
plt.savefig(path_save_figs + "snr_src_1_mix_2.jpeg")


"""
plt.clf()
fig,ax2=plt.subplots()
ax2.hist(np.array(snr_src_1_arr_2_mic),bins=np.arange(100,step=5),color="orange")
ax2.set_ylabel("No of samples")
ax2.set_xlabel("Negative log of variance -log(np.var(sp_signal))")
fig.tight_layout()
plt.savefig("hist_snr_src_1_ARR_2_mic.jpeg")

plt.clf()
fig,ax3=plt.subplots()
ax3.hist(np.array(snr_src_2_arr_2_mic),bins=np.arange(100,step=5),color="orange")
ax3.set_xlabel("SNR dB SRC 2")
plt.savefig("hist_snr_src_2_ARR_2_mic.jpeg")

plt.clf()
fig,ax4=plt.subplots()
ax4.hist(np.array(snr_src_3_arr_2_mic),bins=np.arange(100,step=5),color="orange")
ax4.set_xlabel("SNR dB SRC 3")
plt.savefig("hist_snr_src_3_ARR_2_mic.jpeg")
"""

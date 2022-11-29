import os
import numpy as np
import h5py
import soundfile as sf
from scipy import signal
import random
from audiogen.noise.synthetic import SSN
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io.wavfile import write
from scipy.spatial import distance
import math
import acoustics
import sys

####################################################################################
# Please update all the paths according to your system before launching the script #
####################################################################################

# Train speech data 100000
# root_speech_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/train-clean-360/'

# Speech shape noise data 28000
root_ssn_data = "/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/train-clean-100/"
# Validation data speech
# root_speech_val_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/dev-clean/'
# Test speech data
# root_speech_test_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/test-clean/'

# RIR Data : Simulate RIRs datasets
rir_data_path = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/EM_32/generated_rirs_EM_32_rooms_"


path_speech = "/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/"

# Reverb data
# rir_noise_data_path='/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/generated_rirs_lwh_correct_noise.hdf5'


speech_files = []
num = 0

#Process in batch of 100 rooms each

parallel_batch_no = int(sys.argv[1])

no_of_rooms = 20000
no_of_vps = 3
parallel_jobs = 100
divided_rooms = []
divi_arr = []

for divi in range(no_of_rooms + 1):

    divi_arr.append(divi)
    if divi != 0 and divi % parallel_jobs == 0:
        divi_arr = [j + 20000 for j in divi_arr]
        divided_rooms.append(divi_arr[:-1])
        divi_arr = []
        divi_arr.append(divi)


# np.random.seed(parallel_batch_no*1140)
test_file_15sec = np.load(
    "/home/psrivastava/source_localization/doa_estimation/voicehome2_arr/data_generation/speech_files_5sec.npy"
)
# open speech file
# train_file=np.load('/home/psrivastava/baseline/scripts/pre_processing/speech_files_.npy')
np.random.shuffle(test_file_15sec)

val_file = np.load(
    "/home/psrivastava/baseline/scripts/pre_processing/speech_val_files_.npy"
)
np.random.shuffle(val_file)

test_file = np.load(
    "/home/psrivastava/baseline/scripts/pre_processing/speech_test_files_.npy"
)
np.random.shuffle(test_file)


# open rir file
rir_data = h5py.File(
    rir_data_path
    + str(divided_rooms[parallel_batch_no][0])
    + "_"
    + str(divided_rooms[parallel_batch_no][-1])
    + ".hdf5",
    "r",
)


# Speech shape noise
ssn_no = np.load("/home/psrivastava/baseline/scripts/pre_processing/ssn_files_.npy")
print(ssn_no)

# Random noise for ref signal
no, sr = sf.read("/home/psrivastava/baseline/scripts/pre_processing/rand_sig.flac")

# referense signal ARR1 : Two mic receiver from DIRHA , ARR2 : Three mic receiver from DIRHA.


ref_sig_ch1_ARR1 = np.load(
    "/home/psrivastava/source_localization/doa_estimation/dcase_mic_arr/data_generation/EM_32/ref_signals_noise/EM32_ref_rir_m1_voicehome2.npy"
)
ref_sig_ch2_ARR1 = np.load(
    "/home/psrivastava/source_localization/doa_estimation/dcase_mic_arr/data_generation/EM_32/ref_signals_noise/EM32_ref_rir_m2_voicehome2.npy"
)
filter_ref_sig_ch1_ARR1 = signal.convolve(no, ref_sig_ch1_ARR1, mode="full")
filter_ref_sig_ch2_ARR1 = signal.convolve(no, ref_sig_ch2_ARR1, mode="full")
mean_var_ARR1 = np.var(
    np.concatenate((filter_ref_sig_ch1_ARR1, filter_ref_sig_ch2_ARR1), axis=0)
)


# Randomize room
# room_nos=[i+1 for i in range(20000)]
room_nos = divided_rooms[parallel_batch_no]


# room_nos=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6]
# room_nos=[207,207,207,207,207]
# Randomize view points
# vp_nos=[1,2,3,4,5]*20000
# vp_nos=[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
# vp_nos=[1,2,3,4,5]

no_of_sources = 3
iterate_sources = np.arange(no_of_sources)


# np.random.shuffle(room_nos)
# np.random.shuffle(vp_nos)

adc = []
kbc = []
alc = []
euc_dist = []
# rir_enr=[]
# conv_enr=[]
n = []
sp_var = []
count_file = 0
# Generate 100000 noisy mixtures
# Save file
# test_room=np.arange(18000,20000)

snr_ARR1 = []
snr_ARR2 = []

nf_file = h5py.File(
    "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/EM_32/noisy_mixtures/noisy_mixture_rooms_"
    + str(divided_rooms[parallel_batch_no][0])
    + "_"
    + str(divided_rooms[parallel_batch_no][-1])
    + ".hdf5",
    "w",
)
room_save = nf_file.create_group("room_nos")

print(room_nos)

for i in range(parallel_jobs):  # 19999

    room = room_nos[i]
    n.append(str(room))
    room_id = room_save.create_group("room_" + str(room))

    np_save_mix_ARR2 = np.zeros(
        (3, 2, 32000)
    )  # no_of_sources_in_the_mixture x no_of_mics x length_of_the_signal

    # print("room_no",room)

    rir_len = rir_data["rirs"]["room_" + str(room)]["rirs_length"][0, :]
    rir_noise_len = rir_data["rirs"]["room_" + str(room)]["rirs_noise_length"][0, :]
    array_type = len(rir_data["rirs"]["room_" + str(room)]["rirs_noise_length"][0, :])

    # print(rir_len)
    # print(rir_noise_len)

    for j in iterate_sources:
        if array_type == 2:
            vp = j
            # print((vp-1)*2,(vp*2)-1,room)

            c_ = rir_len[j * 2]
            c_1 = rir_len[(j * 2) + 1]

            ch_1 = rir_data["rirs"]["room_" + str(room)]["rir"][j * 2, :c_]
            ch_2 = rir_data["rirs"]["room_" + str(room)]["rir"][(j * 2) + 1, :c_1]

            # split_sp=train_file[count_file].split('-')

            # Mehcanism to construct the path for the LibriSpeech files.
            split_sp = test_file_15sec[count_file][0].split("-")
            dataset = test_file_15sec[count_file][1]

            sp, srs = sf.read(
                path_speech
                + dataset
                + "/"
                + split_sp[0]
                + "/"
                + split_sp[1]
                + "/"
                + test_file_15sec[count_file][0]
            )

            # sp,srs=sf.read(root_speech_data+"/"+split_sp[0]+"/"+split_sp[1]+"/"+train_file[count_file])

            count_file += 1

            sp_var.append(
                -10 * np.log(np.var(sp))
            )  # sp_var : Shape (no_rooms x mic_array)

            # print(count_file)

            # Just for the sake of expirement i am taking 3 second signal
            # print(sp.shape)
            # print(ch_1.shape)

            reverb_sig_ch1 = signal.convolve(sp, ch_1, mode="full")[48000:80000]
            reverb_sig_ch2 = signal.convolve(sp, ch_2, mode="full")[48000:80000]

            # Reverb After 50 seconds
            if j == 0:
                n_1 = rir_noise_len[0]
                n_2 = rir_noise_len[1]

                laterev_ch1 = rir_data["rirs"]["room_" + str(room)]["rir_noise"][
                    0, :n_1
                ][800:]
                laterev_ch2 = rir_data["rirs"]["room_" + str(room)]["rir_noise"][
                    1, :n_2
                ][800:]

                rand_10_no = [random.randint(0, 28300) for k in range(10)]

                ssn_file = [
                    root_ssn_data
                    + "/"
                    + ssn_no[k].split("-")[0]
                    + "/"
                    + ssn_no[k].split("-")[1]
                    + "/"
                    + ssn_no[k]
                    for k in rand_10_no
                ]

                # Generate speech shape noise
                ssn_obj = SSN(ssn_file)
                ssn_noise = ssn_obj.generate(
                    3 * ssn_obj.target_sr, rnd_gen=np.random.default_rng(123)
                )
                # Convolve with late reverb
                laterev_ch1_ssn = signal.convolve(ssn_noise, laterev_ch1, mode="full")[
                    20:32020
                ]
                # print("laterev",laterev_ch1.shape)
                laterev_ch2_ssn = signal.convolve(ssn_noise, laterev_ch2, mode="full")[
                    20:32020
                ]

                # Calculate alpha and beta
                snr_static = random.randint(60, 70)

                snr_diff = random.randint(35, 60)

                sigma_diff = mean_var_ARR1 / (np.power(10, (snr_diff / 10)))
                sigma_static = mean_var_ARR1 / (np.power(10, (snr_static / 10)))

                # Calculate alpha
                sigma_ssn = np.var(
                    np.concatenate((laterev_ch1_ssn, laterev_ch2_ssn), axis=0)
                )
                alpha = np.sqrt((sigma_diff / sigma_ssn))

                # 2 Channel White noise
                white_noise_ch1 = np.random.normal(0, 1, size=32000)
                white_noise_ch2 = np.random.normal(0, 1, size=32000)

                # Calculate beta
                beta = np.sqrt(sigma_static)

                # fixing alpha and beta
                # alpha_fix=0.001
                # beta_fix=0.001

                static_noise_ch1 = white_noise_ch1 * beta
                static_noise_ch2 = white_noise_ch2 * beta

                # static_noise_ch1_fix=white_noise_ch1*beta_fix
                # static_noise_ch2_fix=white_noise_ch2*beta_fix

                diff_noise_ch1 = laterev_ch1_ssn * alpha
                diff_noise_ch2 = laterev_ch2_ssn * alpha

                # diff_noise_ch1_fix=laterev_ch1_ssn*alpha_fix
                # diff_noise_ch2_fix=laterev_ch2_ssn*alpha_fix

                # signalf_ch1_fix=reverb_sig_ch1+diff_noise_ch1_fix+static_noise_ch1_fix
                # signalf_ch2_fix=reverb_sig_ch2+diff_noise_ch2_fix+static_noise_ch2_fix

                signalf_ch1 = reverb_sig_ch1 + diff_noise_ch1 + static_noise_ch1
                signalf_ch2 = reverb_sig_ch2 + diff_noise_ch2 + static_noise_ch2

                np_save_mix_ARR2[0, 0, :] = signalf_ch1
                np_save_mix_ARR2[0, 1, :] = signalf_ch2

                diff_noise_f = np.concatenate((diff_noise_ch1, diff_noise_ch2), axis=0)

                static_noise_f = np.concatenate(
                    (static_noise_ch1, static_noise_ch2), axis=0
                )

                # snr_src_1=10*np.log10((np.var(np.concatenate((signalf_ch1,signalf_ch2),axis=0)))/(np.var(diff_noise_f)+np.var(static_noise_f)))
                snr_src_1 = 10 * np.log10(
                    (np.var(np.concatenate((reverb_sig_ch1, reverb_sig_ch2), axis=0)))
                    / (np.var(diff_noise_f) + np.var(static_noise_f))
                )

            else:
                eta = math.pow(10, (np.random.uniform(low=-5, high=5) / 20))

                if j == 1:
                    scaled_reverb_sig_ch1_m2 = reverb_sig_ch1 * eta
                    scaled_reverb_sig_ch2_m2 = reverb_sig_ch2 * eta

                    np_save_mix_ARR2[j, 0, :] = (
                        scaled_reverb_sig_ch1_m2 + np_save_mix_ARR2[0, 0, :]
                    )
                    np_save_mix_ARR2[j, 1, :] = (
                        scaled_reverb_sig_ch2_m2 + np_save_mix_ARR2[0, 1, :]
                    )

                elif j == 2:

                    scaled_reverb_sig_ch1_m3 = reverb_sig_ch1 * eta
                    scaled_reverb_sig_ch2_m3 = reverb_sig_ch2 * eta

                    np_save_mix_ARR2[j, 0, :] = (
                        scaled_reverb_sig_ch1_m3 + np_save_mix_ARR2[1, 0, :]
                    )
                    np_save_mix_ARR2[j, 1, :] = (
                        scaled_reverb_sig_ch2_m3 + np_save_mix_ARR2[1, 1, :]
                    )

    SINR_S1 = 10 * np.log10(
        (np.var(np.concatenate((signalf_ch1, signalf_ch2), axis=0)))
        / (
            np.var(
                np.concatenate(
                    (scaled_reverb_sig_ch1_m2, scaled_reverb_sig_ch2_m2), axis=0
                )
            )
            + np.var(
                np.concatenate(
                    (scaled_reverb_sig_ch1_m3, scaled_reverb_sig_ch2_m3), axis=0
                )
            )
            + np.var(diff_noise_f)
            + np.var(static_noise_f)
        )
    )
    SINR_S2 = 10 * np.log10(
        (
            np.var(
                np.concatenate(
                    (scaled_reverb_sig_ch1_m2, scaled_reverb_sig_ch2_m2), axis=0
                )
            )
        )
        / (
            np.var(np.concatenate((signalf_ch1, signalf_ch2), axis=0))
            + np.var(
                np.concatenate(
                    (scaled_reverb_sig_ch1_m3, scaled_reverb_sig_ch2_m3), axis=0
                )
            )
            + np.var(diff_noise_f)
            + np.var(static_noise_f)
        )
    )
    SINR_S3 = 10 * np.log10(
        (
            np.var(
                np.concatenate(
                    (scaled_reverb_sig_ch1_m3, scaled_reverb_sig_ch2_m3), axis=0
                )
            )
        )
        / (
            np.var(np.concatenate((signalf_ch1, signalf_ch2), axis=0))
            + np.var(
                np.concatenate(
                    (scaled_reverb_sig_ch1_m2, scaled_reverb_sig_ch2_m2), axis=0
                )
            )
            + np.var(diff_noise_f)
            + np.var(static_noise_f)
        )
    )
    SINR_AVG = (SINR_S1 + SINR_S2 + SINR_S3) / 3

    snr_ARR2.append([SINR_S1, SINR_S2, SINR_S3])

    if array_type == 2:
        room_id.create_dataset("nsmix_f", (3, 2, 32000), data=np_save_mix_ARR2)
        room_id.create_dataset("nsmix_snr_src_1", 1, data=snr_src_1)
        room_id.create_dataset("SINR_S1", 1, data=SINR_S1)
        room_id.create_dataset("SINR_S2", 1, data=SINR_S2)
        room_id.create_dataset("SINR_S3", 1, data=SINR_S3)
        room_id.create_dataset("SINR_AVG", 1, data=SINR_AVG)

    """
        elif array_type == 2:
            vp=j
        #print((vp-1)*2,(vp*2)-1,room)

            c_=rir_len[j*2]
            c_1=rir_len[(j*2)+1]



            ch_1=rir_data['rirs']['room_'+str(room)]['rir'][j*2,:c_]
            ch_2=rir_data['rirs']['room_'+str(room)]['rir'][(j*2)+1,:c_1]


        #split_sp=train_file[count_file].split('-')

            #Mehcanism to construct the path for the LibriSpeech files.
            split_sp=test_file_15sec[count_file][0].split('-')
            dataset=test_file_15sec[count_file][1]



            sp,srs=sf.read(path_speech+dataset+"/"+split_sp[0]+"/"+split_sp[1]+"/"+test_file_15sec[count_file][0])


            count_file+=1

            sp_var.append([-10*np.log(np.var(sp))]) #sp_var : Shape (no_rooms x mic_array)


            reverb_sig_ch1=signal.convolve(sp,ch_1,mode='full')[20:16020]
            reverb_sig_ch2=signal.convolve(sp,ch_2,mode='full')[20:16020]
            if j==0:
                n_1=rir_noise_len[0]
                n_2=rir_noise_len[1]


                laterev_ch1=rir_data['rirs']['room_'+str(room)]['rir_noise'][0,:n_1][800:]
                laterev_ch2=rir_data['rirs']['room_'+str(room)]['rir_noise'][1,:n_2][800:]


                rand_10_no=[random.randint(0,28300) for k in range(10)]

                ssn_file=[root_ssn_data+"/"+ssn_no[k].split("-")[0]+"/"+ssn_no[k].split("-")[1]+"/"+ssn_no[k] for k in rand_10_no]


                #Generate speech shape noise
                ssn_obj=SSN(ssn_file)
                ssn_noise=ssn_obj.generate(3*ssn_obj.target_sr,rnd_gen=np.random.default_rng(123))
                #Convolve with late reverb
                laterev_ch1_ssn=signal.convolve(ssn_noise,laterev_ch1,mode='full')[20:16020]
                #print("laterev",laterev_ch1.shape)
                laterev_ch2_ssn=signal.convolve(ssn_noise,laterev_ch2,mode='full')[20:16020]



                #Calculate alpha and beta
                snr_static=random.randint(60,70)

                snr_diff=random.randint(25,60)

                sigma_diff=mean_var_ARR1/(np.power(10,(snr_diff/10)))
                sigma_static=mean_var_ARR1/(np.power(10,(snr_static/10)))


                #Calculate alpha
                sigma_ssn=np.var(np.concatenate((laterev_ch1_ssn,laterev_ch2_ssn),axis=0))
                alpha=np.sqrt((sigma_diff/sigma_ssn))


                #2 Channel White noise
                white_noise_ch1=np.random.normal(0,1,size=16000)
                white_noise_ch2=np.random.normal(0,1,size=16000)


                #Calculate beta
                beta=np.sqrt(sigma_static)

                #fixing alpha and beta
                #alpha_fix=0.001
                #beta_fix=0.001

                static_noise_ch1=white_noise_ch1*beta
                static_noise_ch2=white_noise_ch2*beta

                #static_noise_ch1_fix=white_noise_ch1*beta_fix
                #static_noise_ch2_fix=white_noise_ch2*beta_fix

                diff_noise_ch1=laterev_ch1_ssn*alpha
                diff_noise_ch2=laterev_ch2_ssn*alpha

            #diff_noise_ch1_fix=laterev_ch1_ssn*alpha_fix
            #diff_noise_ch2_fix=laterev_ch2_ssn*alpha_fix

            #signalf_ch1_fix=reverb_sig_ch1+diff_noise_ch1_fix+static_noise_ch1_fix
            #signalf_ch2_fix=reverb_sig_ch2+diff_noise_ch2_fix+static_noise_ch2_fix

                signalf_ch1=reverb_sig_ch1+diff_noise_ch1+static_noise_ch1
                signalf_ch2=reverb_sig_ch2+diff_noise_ch2+static_noise_ch2



                np_save_mix_ARR1[0,0,:]=signalf_ch1
                np_save_mix_ARR1[0,1,:]=signalf_ch2

                diff_noise_f=np.concatenate((diff_noise_ch1,diff_noise_ch2),axis=0)

                static_noise_f=np.concatenate((static_noise_ch1,static_noise_ch2),axis=0)

                snr_src_1=10*np.log10((np.var(np.concatenate((signalf_ch1,signalf_ch2),axis=0)))/(np.var(diff_noise_f)+np.var(static_noise_f)))

            else:
                eta=math.pow(10,(np.random.uniform(low=-5,high=5)/10))

                scaled_reverb_sig_ch1=reverb_sig_ch1*eta
                scaled_reverb_sig_ch2=reverb_sig_ch2*eta

                if j == 1:
                    np_save_mix_ARR1[j,0,:]=scaled_reverb_sig_ch1+np_save_mix_ARR1[0,0,:]
                    np_save_mix_ARR1[j,1,:]=scaled_reverb_sig_ch2+np_save_mix_ARR1[0,1,:]

                    snr_static_ch1_ch2_src2=np.var(np.concatenate((scaled_reverb_sig_ch1,scaled_reverb_sig_ch2),axis=0))


                    snr_src_2=10*np.log10((np.var(np.concatenate((np_save_mix_ARR1[j,0,:],np_save_mix_ARR1[j,1,:]),axis=0)))/(np.var(np.concatenate((np_save_mix_ARR1[0,0,:],np_save_mix_ARR1[0,1,:]),axis=0))))


                elif j == 2:
                    np_save_mix_ARR1[j,0,:]=scaled_reverb_sig_ch1+np_save_mix_ARR1[1,0,:]
                    np_save_mix_ARR1[j,1,:]=scaled_reverb_sig_ch2+np_save_mix_ARR1[1,1,:]

                    snr_static_ch1_ch2_src3=np.var(np.concatenate((scaled_reverb_sig_ch1,scaled_reverb_sig_ch2),axis=0))

                    snr_src_3=10*np.log10((np.var(np.concatenate((np_save_mix_ARR1[j,0,:],np_save_mix_ARR1[j,1,:]),axis=0)))/(np.var(np.concatenate((np_save_mix_ARR1[1,0,:],np_save_mix_ARR1[1,1,:]),axis=0))))

                    snr_ARR1.append([snr_src_1,snr_src_2,snr_src_3])
        """

    # kbc.append(snr_fix)
    # rir_enr.append(acoustics.Signal(np.concatenate((ch_1,ch_2),axis=0),16000).energy())
    # conv_enr.append(acoustics.Signal(np.concatenate((reverb_sig_ch1,reverb_sig_ch2),axis=0),16000).energy())

    # print(snr_f)
    # print(snr_f,snr_f2,snr_f3)
    # euc_dist.append(distance.euclidean(rir_data['receiver_config']['room_'+str(room)]['barycenter'][(vp-1),:],rir_data['source_config']['room_'+str(room)]['source_pos'][(vp-1),:]))
    # print("example",i)


"""
        f, Pxx, Sxx=signal.spectrogram(signalf_ch1,16000)
        f2, Pxx_2, Sxx_2=signal.spectrogram(reverb_sig_ch1,16000)
        f3, Pxx_3, Sxx_3=signal.spectrogram(diff_noise_ch1,16000)
        f4, Pxx_4, Sxx_4=signal.spectrogram(ssn_noise,16000)
        #f5, Pxx_5, Sxx_5=signal.spectrogram(ch1, 16000)

        fig=plt.figure()
        gs=gridspec.GridSpec(2,2,wspace=0.5,hspace=0.5)
        ax=plt.subplot(gs[0,0])
        ax1=plt.subplot(gs[0,1])
        ax2=plt.subplot(gs[1,0])
        ax3=plt.subplot(gs[1,1])

        ax.pcolormesh(Pxx,f,np.log10(abs(Sxx)),shading='gouraud')
        ax.set_title("Final Noisy Signal")
        ax1.pcolormesh(Pxx_2,f2,np.log10(abs(Sxx_2)), shading='gouraud')
        ax1.set_title("Reverb Signal")
        ax2.pcolormesh(Pxx_3,f3,np.log10(abs(Sxx_3)), shading='gouraud')
        ax2.set_title("Diffuse Noise")
        ax3.pcolormesh(Pxx_4,f4,np.log10(abs(Sxx_4)), shading='gouraud')
        ax3.set_title("Speech Shape Noise")



        #plt.semilogy(f,np.sqrt(Pxx),label="Noisy Mixture")
        #plt.semilogy(f2,np.sqrt(Pxx_2),c='y',label="Clean Rev Signal")
        #plt.xlabel("Frequency")
        #plt.ylabel("Linear Spectrum [V RMS]")
        #plt.legend()
        #print("reverb signal",np.sqrt(Pxx_2.max()))


        #plt.semilogy(f,np.sqrt(Pxx))

        fig.add_subplot(ax)
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
        fig.add_subplot(ax3)


        fig.savefig("psd_noisy_"+str(i)+".jpeg")

        fig.clf()
        #plt.semilogy(f2, np.sqrt(Pxx_2))
        #plt.savefig("rever_"+str(i)+".jpeg")

        #plt.plot(np.arange(48000),signalf_ch1)
        #plt.savefig("ref_noise_"+str(i)+".jpeg")
        """

# write("ns_mix_"+str(room)+str(i+1)+".wav",16000,abc)
# sf.write("ns_mix_"+"ch_2_"+str(i)+".flac",signalf_ch2,16000)
# sf.write("ns_simple_"+str(i)+".flac",reverb_sig_ch1,16000)


# fig,ax1=plt.subplots()
# cs=['b','b','b','b','b','r','r','r','r','r','g','g','g','g','g','y','y','y','y','y','orange','orange','orange','orange','orange','violet','violet','violet','violet','violet']
# for l in range(30):


# ax1.scatter(euc_dist,adc,color='b')

# for i,txt in enumerate(n):
#    ax1.annotate(txt,(euc_dist[i],adc[i]))


# ax1.set_ylabel('SNR dB',c='b')
# ax1.set_xlabel('Euclidian Distance',c='b')

# print(euc_dist)
# print(adc)

"""
ax2=ax1.twinx()
ax2.plot(np.arange(10),euc_dist,c='r',marker='s',label="Euclidian Distance")
ax2.set_ylabel('Euc distance',c='r')
"""

# fig.tight_layout()
# plt.legend()
# plt.savefig("/home/psrivastava/baseline/scripts/haikus_project/snr_plot_fix_snr_static_room_200samples.jpeg")
# fig.clf()

print("count_file", count_file)
fig, ax2 = plt.subplots()
ax2.hist(np.array(snr_ARR2)[:, 0])
ax2.set_ylabel("No of samples")
ax2.set_xlabel("SINR In dB ")
fig.tight_layout()
plt.savefig("SINR_src_1.jpeg")

fig, ax3 = plt.subplots()
ax3.hist(np.array(snr_ARR2)[:, 1])
ax3.set_xlabel("SINR In dB Src 2")
plt.savefig("SINR_src_2.jpeg")

fig, ax4 = plt.subplots()
ax4.hist(np.array(snr_ARR2)[:, 2])
ax4.set_xlabel("SINR In dB Src 3")
plt.savefig("SINR_src_3.jpeg")


fig, ax4 = plt.subplots()
ax4.hist(np.mean(np.array(snr_ARR2), axis=1))
ax4.set_xlabel("SINR In dB Src 3")
plt.savefig("SINR_src_avg.jpeg")


"""
fig2,ax2=plt.subplots()

for n in range(30):
    ax2.scatter(euc_dist[n],kbc[n],color=cs[n])
ax2.set_ylabel('SNR dB',c='b')
ax2.set_xlabel('Euclidian Distance',c='b')
print(euc_dist)
print(kbc)
fig2.tight_layout()
plt.legend()
plt.savefig("/home/psrivastava/baseline/scripts/haikus_project/snr_plot_fix_alpha_beta_room.jpeg")

fig3,ax3=plt.subplots()
for n in range(30):
    ax3.scatter(euc_dist[n],rir_enr[n],color=cs[n])
ax3.set_ylabel('Energy',c='b')
ax3.set_xlabel('Euclidian Distance',c='b')
fig3.tight_layout()
plt.legend()
plt.savefig("/home/psrivastava/baseline/scripts/haikus_project/energy_decay_rir.jpeg")

fig4,ax4=plt.subplots()
for n in range(30):
    ax4.scatter(euc_dist[n],conv_enr[n],color=cs[n])
ax4.set_ylabel("Energy",c='b')
ax4.set_xlabel('Euclidian Distance',c='b')
fig4.tight_layout()
plt.legend()
plt.savefig("/home/psrivastava/baseline/scripts/haikus_project/energy_decay_reverbspeech.jpeg")


"""


"""
for i in os.listdir(root_speech_test_data):
    #speech_files[i]={j:[] for j in os.listdir(root_ssn_data+"/"+i)}
    for j in os.listdir(root_speech_test_data+"/"+i):
        #js=[]
        for k in os.listdir(root_speech_test_data+"/"+i+"/"+j):
            if ".flac" in k:
                data,sr=sf.read(root_speech_test_data+"/"+i+"/"+j+"/"+k)
                if data.shape[0] > (sr*2 + 1000):
                    speech_files.append(k)
                    num+=1
        #speech_files[i][j]=js



np.save("speech_test_files_.npy",speech_files)
#print(speech_files)
print(num)
"""

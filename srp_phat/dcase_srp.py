#############################################################
# Calculate GCC-PHAT on dcase and other simulated datasets  #
#############################################################

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
import yaml
from yaml.loader import SafeLoader

def polar2cart(phi, theta, r):
    phi=np.deg2rad(phi)
    theta=np.deg2rad(theta)

    return [
         r * math.cos(theta) * math.cos(phi),
         r * math.cos(theta) * math.sin(phi),
         r * math.sin(theta)
    ]



number_of_sources=1
fs=16000
nfft=1536
tolerance=5
d=0.104
phi=0
M=2
pred_gt=[]
c=343.0
freq_bins = np.arange(30,1024)


########################################################
#             DCASE MIC ANNOTATIONS                    #
########################################################
voicehome2_mic_1_=np.array([0.037, 0.056, -0.038])
voicehome2_mic_2_=np.array([-0.034, 0.056, 0.038])

dcase2_mic_1_=np.array(polar2cart(45,35,4.2))
dcase2_mic_2_=np.array(polar2cart(-45,-35,4.2))
dcase2_mic_3_=np.array(polar2cart(135,-35,4.2))
dcase2_mic_4_=np.array(polar2cart(-135,35,4.2))


distance_between_mic=distance.euclidean(dcase2_mic_1_,dcase2_mic_2_)
print(distance_between_mic)
barycenter_mic=(dcase2_mic_1_+dcase2_mic_2_)/2
u_vec=barycenter_mic-dcase2_mic_2_






pred_gt=[]
vh_mic=np.array([voicehome2_mic_1_,voicehome2_mic_2_]).T



acc=0
err=0
co=0
all_doa=[]
all_tdoa=[]
speed_of_sound=343

##############################################################
#      DCASE REAL DATASET DATA AND DOA'S                     #
##############################################################
'''
path="/home/psrivastava/source_localization/doa_estimation/dcase_mic_arr/test_scripts/"
data=np.load(path+"data_dcase_24k_2sec.npy")
coord=np.load(path+"coord_dcase_24k_2sec.npy")
print(data.shape)
print(coord.shape)
'''
#"data_dcase_24k_2sec.npy"
#""coord_dcase_24k_2sec.npy""
##############################################################





###########################################################
# RESAMPLE DCASE REAL DATASET THAT IS IN 24KHZ TO 16 KHZ  #
###########################################################

number_of_samples = round(48000 * 16/24)


############################################################
# CHECK GCC-PHAT ON OTHER DATASETS                         #
############################################################


data=[]
coord=[]

path_em32="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/voicehome2_arr/D5_1101/noisy_mixtures/"
nsmix=h5py.File(path_em32+"D5_1101_aggregated_mixture.hdf5")
anno_mix=h5py.File(path_em32+"D5_1101_aggregated_mixture_annotations.hdf5")

#with open('/home/psrivastava/source_localization/doa_estimation/dcase_mic_arr/data_generation/EM_32/debug_data/conf_source_EM32.yml') as f_2:
#    source_rooms = yaml.load(f_2, Loader=SafeLoader)

mic_7=np.array([1.5051453880902617, 3.9210377912882475, 2.571758278209442e-16])/100
mic_11=np.array([1.5051453880902617, -3.9210377912882475, 2.571758278209442e-16])/100

mic_1=np.array([0.037, 0.056, -0.038])
mic_2=np.array([-0.034, 0.056, 0.038])

#mic_6=np.array([0.02432757,0.02432757,0.02409021])
#mic_10=np.array([0.02432757, -0.02432757, -0.02409021])

#mic_bc=(mic_6+mic_10)/2
#u_vec=mic_bc-mic_6



######################################################################################
# distance_between_mic in real datasets                                              #
# vh : 0.104 = 10.4 cm                                                               #
# dcase : Varies = 6.8 b/w (Mic 6 {5}, Mic 10 {9}) 7.8cm b/w (Mic 7 {6}, Mic 11 {10})#
# DIRHA : 0.30 = 30 cm                                                               #
######################################################################################


distance_between_mic=euclidean(mic_1,mic_2)*100
#distance_between_mic=30.0
#print(distance_between_mic)


for i in range(25000,27000):
    data.append(nsmix["room_nos"]["room_"+str(i)]["nsmix_f"][0,:,:])
    coord.append(anno_mix["room_nos"]["room_"+str(i)]["azimuth"][()][0])

    #########################
    #Recalculate DOA's code #
    #########################

    #rec_pos=anno_mix["room_nos"]["room_"+str(i)]["rec_pos"][()][0,:]
    #barycenter=anno_mix["room_nos"]["room_"+str(i)]["barycenter"][()][0,:]
    #new_mic_7=barycenter+mic_7

    #new_mic_11=barycenter+mic_11

    #new_mic_bc=(new_mic_7+new_mic_11)/2


    #src_pos=np.array(source_rooms["room_"+str(i)]["source_pos"][0])

    #u_vec=new_mic_bc-new_mic_7
    #v_vec=new_mic_bc-src_pos

    #azimuth_src=np.rad2deg(np.arccos(np.dot(u_vec,v_vec)/(np.sqrt(np.sum(u_vec**2))*np.sqrt(np.sum(v_vec**2)))))
    #coord.append(azimuth_src)


data=np.array(data)
coord=np.array(coord)


for b in np.arange(data.shape[0]):


    #Transpose if the mics are present on the axis=-1

    #data_16k=data[b,:,:2].T #Use For dcase

    data_16k=data[b,:,:]

    # Resample dcase dataset #

    #data_16k=signal.resample(data_16k,number_of_samples,axis=-1)


    #v_vec=coord[b,:] #Use for dcase

    #azimuth_src=np.rad2deg(np.arccos(np.dot(u_vec,v_vec)/(np.sqrt(np.sum(u_vec**2))*np.sqrt(np.sum(v_vec**2)))))  #Use for dcase

    azimuth_src=coord[b]


    delay=pra.experimental.localization.tdoa(data_16k[0,:],data_16k[1,:],distance_between_mic/100,interp=4,fs=16000)

    #if delay > -0.0008:
    all_tdoa.append([azimuth_src,delay])

    estimated_angle=np.abs(180-np.rad2deg(np.arccos((delay*speed_of_sound)/(distance_between_mic/100))))   #For DCASE and voice home use this
    #estimated_angle=np.rad2deg(np.arccos((delay*speed_of_sound)/(distance_between_mic/100)))   #For DIRHA use this and DCASE

    if not np.isnan(estimated_angle):
        angular_err=np.abs(azimuth_src-estimated_angle)

        err+=angular_err

        if angular_err <=10:
            acc+=1

        all_doa.append([azimuth_src,estimated_angle])

            #print(delay)
            #wavfile.write("-2_delay.wav",24000,data_16k[0,:])
            #break

        co+=1



#################################################
# Calculate confidence interval and other stats #
#################################################

import math
all_tdoa=np.array(all_tdoa)
all_doa=np.array(all_doa)
plt.scatter(all_tdoa[:,0],all_tdoa[:,1])
#plt.hist(all_doa,bins=np.linspace(0,180,num=30),color="blue")
#plt.savefig("all_tdoa_dcase_decimate_sim_correct.jpeg")
errs=np.abs(all_doa[:,0]-all_doa[:,1])
mean=np.mean(errs)
std=np.std(errs)

ci_95=1.96*std/math.sqrt(all_doa.shape[0])

plt.clf()
plt.scatter(all_doa[:,0],all_doa[:,1])

plt.xlabel("Ground Truth DOA")
plt.ylabel("Estimated DOA")
plt.plot(np.linspace(0,180,num=360),np.linspace(0,180,num=360),color="orange")
#plt.savefig("err_decimate_sim_correct.jpeg")

print("Accuracy over validation set SRP-PHAT",(acc/co)*100)
print("Mean angular error over validation set SRP-PHAT",(err/co))
print("CI_95",ci_95)

########################################
# Retreive DIRHA data from the dataset #
########################################

import numpy as np
import os
from scipy.io import wavfile
import math
import xml.etree.ElementTree as ET


def polar2cart(phi, theta, r=1):
    phi=np.deg2rad(phi)
    theta=np.deg2rad(theta)

    return [
         r * math.cos(theta) * math.cos(phi),
         r * math.cos(theta) * math.sin(phi),
         r * math.sin(theta)
    ]








path="/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/DIRHA_English_WSJ/data/DIRHA_English_wsj/Real/"


data_audio=[]
data_cord=[]
record_=[]
for folders in os.listdir(path):

    #Construct wav file path

    wavfile_path_L=path+folders+"/"+"Livingroom"+"/"+"Wall"+"/"+"L4L.wav"
    wavfile_path_R=path+folders+"/"+"Livingroom"+"/"+"Wall"+"/"+"L4R.wav"

    s,audio_file_L=wavfile.read(wavfile_path_L)
    s,audio_file_R=wavfile.read(wavfile_path_R)

    # Construct annotation file path

    xml_path_L=path+folders+"/"+"Livingroom"+"/"+"Wall"+"/"+"L4L.xml"
    xml_path_R=path+folders+"/"+"Livingroom"+"/"+"Wall"+"/"+"L4R.xml"

    tree_R=ET.parse(xml_path_R).getroot()
    tree_L=ET.parse(xml_path_L).getroot()

    mic_pos_1=np.zeros(3)
    mic_pos_2=np.zeros(3)

    # Retreive mic positions

    mic_pos_1[0]=float(tree_R[0][3].text.split(";")[0].split("=")[1])/100
    mic_pos_1[1]=float(tree_R[0][3].text.split(";")[1].split("=")[1])/100
    mic_pos_1[2]=float(tree_R[0][3].text.split(";")[2].split("=")[1])/100

    mic_pos_2[0]=float(tree_L[0][3].text.split(";")[0].split("=")[1])/100
    mic_pos_2[1]=float(tree_L[0][3].text.split(";")[1].split("=")[1])/100
    mic_pos_2[2]=float(tree_L[0][3].text.split(";")[2].split("=")[1])/100

    print(mic_pos_1)
    print(mic_pos_2)

    mic_bc=(mic_pos_1+mic_pos_2)/2
    u_vec=mic_bc-mic_pos_1
    print(mic_bc)
    print(folders)
    #Speakers in the wavfile.

    for j in np.arange(len(tree_R))[1:]:

        # Retreive speaker positions

        spk_pos=np.zeros(3)
        audio_data=np.zeros((2,32000))
        spk_pos[0]=float(tree_R[j][3].text.split(" ")[0].split("=")[1])/100
        spk_pos[1]=float(tree_R[j][3].text.split(" ")[1].split("=")[1])/100
        spk_pos[2]=float(tree_R[j][3].text.split(" ")[2].split("=")[1])/100


        v_vec=mic_bc-spk_pos


        # Calculate GT DoA

        azimuth_src=np.rad2deg(np.arccos(np.dot(u_vec,v_vec)/(np.sqrt(np.sum(u_vec**2))*np.sqrt(np.sum(v_vec**2)))))

        record_.append((folders,j))

        begin_sample=int(tree_R[j][4].text)
        #print(audio_file_L[begin_sample:(begin_sample+32000)].shape[0])

        #Retreive 2 second audio samples from 2 channels.

        audio_data[0,:]=audio_file_L[begin_sample:(begin_sample+32000)]
        audio_data[1,:]=audio_file_R[begin_sample:(begin_sample+32000)]

        data_audio.append(audio_data)
        data_cord.append([azimuth_src])


np.save("data_DIRHA_3.npy",np.array(data_audio))
np.save("DOA_DIRHA_3.npy",np.array(data_cord))
np.save("record_data_3.npy",np.array(record_))

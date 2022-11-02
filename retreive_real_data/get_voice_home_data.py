############################################################
# Retreive voiceHome2 data to test for the testing purpose #
############################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


audio_path="/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/voiceHome2/orig_corpora/audio/noisy/"
annotation_path="/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/voiceHome2/orig_corpora/annotations/rooms/"

nHouse = 4  # number of houses

nRoom = 3  # number of rooms per house

nSpk = 3  # number of speakers per house

nPos = 5  # number of speakers positions

nNoise = 4  # number of noise conditions per room

nUtt = 2  # number of utterances per {spk,pos,room,house,noise}



house_1={'room_1':{},'room_2':{},'room_3':{}}   #[{'quite':[],'noiseCond1':[],'noiseCond2':[],'noiseCond3':[]}, {'quite':[],'noiseCond1':[],'noiseCond2':[],'noiseCond3':[]} , {'quite':[],'noiseCond1':[],'noiseCond2':[],'noiseCond3':[]}]
house_2={'room_1':{},'room_2':{},'room_3':{}}
house_3={'room_1':{},'room_2':{},'room_3':{}}
house_4={'room_1':{},'room_2':{},'room_3':{}}
hc=[house_1,house_2,house_3,house_4]

house_1_ano={'room_1':{},'room_2':{},'room_3':{}}   #[{'quite':[],'noiseCond1':[],'noiseCond2':[],'noiseCond3':[]}, {'quite':[],'noiseCond1':[],'noiseCond2':[],'noiseCond3':[]} , {'quite':[],'noiseCond1':[],'noiseCond2':[],'noiseCond3':[]}]
house_2_ano={'room_1':{},'room_2':{},'room_3':{}}
house_3_ano={'room_1':{},'room_2':{},'room_3':{}}
house_4_ano={'room_1':{},'room_2':{},'room_3':{}}
hc_ano=[house_1_ano,house_2_ano,house_3_ano,house_4_ano]



enc=[['F1','M1','M2'],['F2','M3','M4'],['F3','M5','M6'],['F4','M7','M8']] #Encoding of Female and Male speakers in 4 different houses.
noise_cond_arr=[]
c=1
for noise_cond in np.arange(12):
    noise_cond_arr.append([str('1'),str(c+1),str(c+2),str(c+3)])
    c=c+3

print(noise_cond_arr)

dir_objs=os.listdir(audio_path)
cp=0
r_=0

voicehome2_mic_1_=np.array([0.037, 0.056, -0.038])
voicehome2_mic_2_=np.array([-0.034, 0.056, 0.038])

#voicehome2_mic_1_=np.array([-0.052, 0, 0])
#voicehome2_mic_2_=np.array([0.052, 0, 0])


for h_no in np.arange(nHouse): #Iterate through homes
    print("house_no",h_no)
    for r_no in np.arange(nRoom):  # Iterate through rooms, each home has 3 rooms.

        # Construct path to retreive position file.
        array_pos_file="home"+str(h_no+1)+"_"+"room"+str(r_no+1)+"_"+"arrayPos1"+".txt"

        # Open file
        f_open=open(annotation_path+array_pos_file,'r')
        str_coord=f_open.readlines()

        print(str_coord)

        # Get barycenter poistion of the mic array in the room
        array_cord_bc=np.array([float(z) for z in str_coord[0].split("\t")[1:4]])

        # Shift mic positions towards the barycenter position in the room

        voicehome2_mic_1=voicehome2_mic_1_+array_cord_bc
        voicehome2_mic_2=voicehome2_mic_2_+array_cord_bc

        # Calculate barycenter of the "2 mic array", that is on one of the side panel

        bc_mic=(voicehome2_mic_1+voicehome2_mic_2)/2
        u_vec=voicehome2_mic_2-bc_mic

        for spk_no in np.arange(nSpk): #Iterate through number of speakers, each home has 3 speakers
            print("speaker",spk_no)
            hc[h_no]['room_'+str(r_no+1)][enc[h_no][spk_no]]={}
            hc_ano[h_no]['room_'+str(r_no+1)][enc[h_no][spk_no]]={}

            for pos_no in np.arange(nPos): #Iterate through each speaker positions,each speaker has 5 position.
                    print("position",pos_no+1)

                    #Construct the path to retreive wav file and the anno file describing the speaker positions.
                    wav_file_name="home"+str(h_no+1)+"_"+"room"+str(r_no+1)+"_"+"arrayGeo1"+"_"+"arrayPos1"+"_"+"speaker"+enc[h_no][spk_no]+"_"+"speakerPos"+str(pos_no+1)+"_"+"noiseCond"
                    anno_file_name="home"+str(h_no+1)+"_"+"room"+str(r_no+1)+"_"+"speaker"+enc[h_no][spk_no]+"_"+"speakerPos"+str(pos_no+1)+".txt"
                    f_open_=open(annotation_path+anno_file_name,'r')
                    str_coord_spk=f_open_.readlines()
                    print(str_coord_spk)
                    # Retreive speaker positions
                    str_coord_spk=str_coord_spk[0].split("\t")
                    spk_cord=np.array([float(z) for z in str_coord_spk[1:4]])
                    spk_dir=[float(str_coord_spk[-2]),float(str_coord_spk[-1])]
                    v_vec=spk_cord-bc_mic

                    # Caclulate GT DoA 
                    doa_angle=np.rad2deg(np.arccos(np.dot(u_vec,v_vec)/(np.sqrt(np.sum(u_vec**2))*np.sqrt(np.sum(v_vec**2)))))

                    #'bc_mic':bc_mic,
                    hc_ano[h_no]['room_'+str(r_no+1)][enc[h_no][spk_no]]['position_'+str(pos_no+1)]={'doa_angle':doa_angle,'array_cord_bc':array_cord_bc,'spk_cord':spk_cord,'spk_dir':spk_dir,'mic_1':voicehome2_mic_1,'mic_2':voicehome2_mic_2}

                    c_=[]
                    for j in dir_objs:
                        if wav_file_name in j:
                            cp+=1
                            #samplerate, data = wavfile.read(audio_path+j)
                            c_.append(j)

                    hc[h_no]['room_'+str(r_no+1)][enc[h_no][spk_no]]['position_'+str(pos_no+1)]=c_


print(cp)

np.save("voiceHome2_house_1.npy",house_1)
np.save("voiceHome2_house_2.npy",house_2)
np.save("voiceHome2_house_3.npy",house_3)
np.save("voiceHome2_house_4.npy",house_4)

np.save("voiceHome2_house_1_ano.npy",house_1_ano)
np.save("voiceHome2_house_2_ano.npy",house_2_ano)
np.save("voiceHome2_house_3_ano.npy",house_3_ano)
np.save("voiceHome2_house_4_ano.npy",house_4_ano)


'''
print("xxxxxxxxxxxxxxxxxxxxxxx")
for ncond in np.arange(4):
    nc=noise_cond_arr[r_][ncond]
    print("ncond",nc)

    wav_file_name="home"+str(h_no+1)+"_"+"room"+str(r_no+1)+"_"+"arrayGeo1"+"_"+"arrayPos1"+"_"+"speaker"+enc[h_no][spk_no]+"_"+"speakerPos"+str(pos_no+1)+"_"+"noiseCond"+str(nc)

    c_=[]
    for j in dir_objs:
        if wav_file_name in j:
            cp+=1
            #samplerate, data = wavfile.read(audio_path+j)
            c_.append(j)

    hc[h_no]['room_'+str(r_no+1)][enc[h_no][spk_no]]['position_'+str(pos_no+1)]['noiseCond_'+str(nc)]=c_

r_+=1
'''

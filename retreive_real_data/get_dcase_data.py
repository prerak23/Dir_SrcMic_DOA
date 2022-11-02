#######################################################################################################
#   Get STARSS22 dataset for testing purpose, we focus on audio parts where only one source is active #
#######################################################################################################

import numpy as np
import os
from scipy.io import wavfile
import math

def polar2cart(phi, theta, r=1):
    phi=np.deg2rad(phi)
    theta=np.deg2rad(theta)

    return [
         r * math.cos(theta) * math.cos(phi),
         r * math.cos(theta) * math.sin(phi),
         r * math.sin(theta)
    ]






path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/real_data/metadata_dev/"
path_audio="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/real_data/mic_dev/"

c=0
path_test="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/real_data/metadata_dev/dev-train-sony/fold3_room21_mix028.csv"

data_audio=[]
data_cord=[]

##############################################
# Brute force algorithm to retreive the data #
##############################################

for folders in os.listdir(path):
    for files in os.listdir(path+folders):

        abc=np.genfromtxt(path+folders+"/"+files, delimiter=',')

        frames=abc[:,0]


        no_repeated_frames=[]

        for idx,f in enumerate(frames):
            no_repeat=False

            for idx_2,j in enumerate(frames):
                if f == j and idx_2 != idx :
                    no_repeat=True

            if no_repeat == False:
                no_repeated_frames.append(f)

        print(path+folders+"/"+files)
        no_repeated_frames=np.array(no_repeated_frames)
        k=[]
        track_=np.array([False]*no_repeated_frames.shape[0])
        #print(no_repeated_frames)

        if no_repeated_frames.shape[0] > 20:
            for idx,j in enumerate(no_repeated_frames):
                local_track=[]
                if track_[idx] == False and (idx+20)<= no_repeated_frames.shape[0]:
                    for r in np.arange(20):
                        frame_questioned=no_repeated_frames[idx+r]
                        if j+r == frame_questioned:
                            local_track.append(frame_questioned)
                            if len(local_track) == 20:
                                track_[idx:idx+20]=True
                                k.append((j,local_track[-1]))
                                local_track=[]
                            else:
                                track_[idx:idx+len(local_track)]=True
            #print(k)

            if len(k) > 0:
                for j in k:
                    if (abc[np.where(abc[:,0]==j[0])[0],1] == 0 or abc[np.where(abc[:,0]==j[0])[0],1] == 1) and (abc[np.where(abc[:,0]==j[1])[0],1] == 0 or abc[np.where(abc[:,0]==j[1])[0],1] == 1):
                        file_wav=files.split(".")[0]
                        s,data=wavfile.read(path_audio+folders+"/"+file_wav+".wav")
                        data_audio.append(data[int(j[0])*2400:(int(j[1])+1)*2400,:])
                        data_cord.append(polar2cart(int(abc[np.where(abc[:,0]==j[0])[0],3]),int(abc[np.where(abc[:,0]==j[0])[0],4])))


np.save("data_dcase_24k_2sec.npy",np.array(data_audio))
np.save("coord_dcase_24k_2sec.npy",np.array(data_cord))






#print(track_)
#print(no_repeated_frames)




#print("---------------")
#print(k)
#k=np.array(k)


#for j in k:
#    print(j[0],abc[np.where(abc[:,0]==j[0])[0],:])
#    print(j[1],abc[np.where(abc[:,0]==j[1])[0],:])

#for idx,c in enumerate(k[:-1]):
#    if (k[idx+1][0]-c[1])>=20:
#            op.append(c)



#print(k)

#if len(np.where(no_repeated_frames==920)[0]) > 0:
#    print(np.where(no_repeated_frames==920)[0])






'''
for folders in os.listdir(path):
    for files in os.listdir(path+folders):
        abc=np.genfromtxt(path+folders+"/"+files, delimiter=',')
        frames=abc[:,0]
        no_repeated_frames=[]

        for f in frames:
            no_repeat=False
            for j in frames:

                if f == j:
                    no_repeat=True

            if no_repeat == False:
                no_repeated_frames.append(f)


        print(path+folders+"/"+files)
        print("--------------------")
        print(no_repeated_frames)

        c+=1
'''

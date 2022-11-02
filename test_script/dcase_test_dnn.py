##################################################################
#  Calculate performance of the model, over DIRHA corpus         #
##################################################################

import numpy as np
import torch
import h5py
#import data_loader_test
import he_cnn_resnet as net_

import torch.nn as nn
import torch.optim as optim

from scipy.io import wavfile
from scipy import signal
import math
from scipy.spatial import distance
import os

device=torch.device("cuda")
net = net_.he_archs(input_size=(4,33,1025)).to(torch.device(device))
path_="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/training/D6/train_again/"

###################################################################
# Iterate through the list of checkpoints in this particular path #
###################################################################

for o_ in os.listdir(path_):
    if ".pt" in o_:

        chkp=torch.load(path_+o_,map_location=device)
        print(o_)
        net.load_state_dict(chkp['model_dict'])
        optimizer = optim.Adam(net.parameters(), lr=0.0001,weight_decay=1e-5)
        optimizer.load_state_dict(chkp['optimizer_dic'])

        net.eval()

        window_len=2048 #128 ms in 1 sec length of signal
        time_segment_stft=33 # For 1 second of signal and 50 % overlap
        max_channel_data=2
        device="cpu"
        ACT_NONE = 0
        _STAGE1_LOSS = nn.MSELoss()
        batch_size=64
        no_of_epochs=110
        sigmoid= nn.Sigmoid()

        ######################
        # Calculate features #
        ######################

        def cal_features(data):


            data_stft_real=np.zeros((data.shape[0],max_channel_data,window_len//2+1,time_segment_stft))
            data_stft_imag=np.zeros((data.shape[0],max_channel_data,window_len//2+1,time_segment_stft))


            for batch in np.arange(data.shape[0]):
                data_stft=signal.stft(data[batch,:,:],axis=-1,fs=16000,nperseg=window_len)

                data_stft_real[batch,:,:,:]=np.real(data_stft[2])
                data_stft_imag[batch,:,:,:]=np.imag(data_stft[2])

            x=np.concatenate((data_stft_real,data_stft_imag),axis=1)
            x=np.transpose(x,(0,1,3,2)) # Dimension (Batch_size x 4 {2 real and 2 imag coeffs} x time_frames x freq_bins )

            return torch.tensor(x).to(device=device)


        def polar2cart(phi, theta, r):
            phi=np.deg2rad(phi)
            theta=np.deg2rad(theta)

            return [
                 r * math.cos(theta) * math.cos(phi),
                 r * math.cos(theta) * math.sin(phi),
                 r * math.sin(theta)
            ]



        ##################################
        # Load DCASE Data And Annotation #
        ##################################

        path="/home/psrivastava/source_localization/doa_estimation/dcase_mic_arr/test_scripts/"
        data=np.load(path+"data_dcase_24k_2sec.npy")
        coord=np.load(path+"coord_dcase_24k_2sec.npy")

        ################################################
        # Eigen mike positions in spherical coordinates#
        ################################################

        dcase2_mic_1_=np.array(polar2cart(45,35,0.042)) # Mic 6
        dcase2_mic_2_=np.array(polar2cart(-45,-35,0.042)) # Mic 10
        dcase2_mic_3_=np.array(polar2cart(135,-35,0.042))
        dcase2_mic_4_=np.array(polar2cart(-135,35,0.042))


        distance_between_mic=distance.euclidean(dcase2_mic_1_,dcase2_mic_2_)
        print(distance_between_mic)
        barycenter_mic=(dcase2_mic_1_+dcase2_mic_2_)/2
        u_vec=barycenter_mic-dcase2_mic_2_



        number_of_samples = round(48000 * 16/24)

        speed_of_sound=343

        data_16k_arr=np.zeros((data.shape[0],2,32000))

        doa_16k=[]

        ##################################
        # Iterate through the DCASE data #
        ##################################

        for b in np.arange(data.shape[0]):

            data_16k=data[b,:,:2].T
            #print(b)

            ##################################
            # Resample datat from 24k to 16k #
            ##################################

            data_16k=signal.resample(data_16k,number_of_samples,axis=-1)

            data_16k_arr[b,:,:]=data_16k

            ##########################
            # GT Pos. Target spekaer #
            ##########################

            v_vec=coord[b,:]

            ##########################
            # Re-calculate, DOA      #
            ##########################

            azimuth_src=np.rad2deg(np.arccos(np.dot(u_vec,v_vec)/(np.sqrt(np.sum(u_vec**2))*np.sqrt(np.sum(v_vec**2)))))

            doa_16k.append([azimuth_src])


        doa_16k=np.array(doa_16k)
        import matplotlib.pyplot as plt
        stft_feat=cal_features(data_16k_arr)

        output_spectrum_stack=np.zeros((stft_feat.shape[0],360))
        #print(stft_feat.shape)
        for j in np.arange(stft_feat.shape[0]):
            data_unsq=torch.unsqueeze(stft_feat[j,:,:,:],axis=0)
            #print(data_unsq.shape,j)
            output_spectrum=net(data_unsq.cuda().float())
            output_spectrum_stack[j,:]=output_spectrum.detach().cpu().numpy()


        
        acc,err,c,d=net_.doa_metric(output_spectrum_stack,doa_16k,"test")
        c=np.array(c)

        errs=np.abs(c[:,0]-c[:,1])
        mean=np.mean(errs)
        std=np.std(errs)

        ci_95=1.96*std/math.sqrt(doa_16k.shape[0])

        plt.scatter(c[:,1],c[:,0])

        plt.plot(np.linspace(0,180,num=360),np.linspace(0,180,num=360),color="orange")
        plt.ylabel("Predicted")
        plt.xlabel("GT DoA")
        plt.savefig("scatter_vc.test_dcase.jpeg")
        print("Accuracy ",acc)
        print("Angular Error ",err)
        print("Confidence Interval 95",mean,ci_95)

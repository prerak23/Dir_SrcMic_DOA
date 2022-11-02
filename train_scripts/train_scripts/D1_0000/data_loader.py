#Data loader for pytorch architecture
#####################################################################################################################
# This dataloader can be change to support:                                                                         #
# More number of sources in the mixture : no_of_sources, elevation, azimuth, elevation_spectrum, azimuth_spectrum   #
# Different type of arrays structures in the room : channel_in_current_data, noisy_mixture_all_channels_            #
# ###################################################################################################################




import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import h5py
import yaml
import matplotlib.pyplot as plt
import soundfile as sf
#from asteroid.filterbanks import STFTFB
#from asteroid.filterbanks.enc_dec import Filterbank, Encoder, Decoder

dataset_name="D1_0000"
path_root="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/dcase_arr/"
abcd=h5py.File(path_root+dataset_name+"/noisy_mixtures/"+dataset_name+"_aggregated_mixture.hdf5",'r') #Open the noisy mixture file
anno_file=h5py.File(path_root+dataset_name+"/noisy_mixtures/"+dataset_name+"_aggregated_mixture_annotations.hdf5",'r')


max_no_of_channels_in_dataset=2 #Currently the dataset support 2 channels from a Mic ARRAY both voicehome2_arr and dcase uses 2 were experimented with 2 channels
length_of_mixture=32000 # 1 second signal

#rt60_file=h5py.File('rt60_anno_room_20_median.hdf5','r') #Open the annotation file
#anno_file=h5py.File('absorption_surface_calcul.hdf5','r') #Open the annotation file

class binuaral_dataset(Dataset):
    def __init__(self,randomized_arr):

        self.random_arr=np.load(randomized_arr) #Randomize the room_number and view points , therefore [(Room_0,vp_1),(Room_2,vp_4),(Room_3,vp_5),(Room_2,vp_3)]


    def __len__(self):
        return len(self.random_arr)

    def __getitem__(self, item_):
        item=self.random_arr[item_] #Get room number
        #print(item)
        no_of_sources=int(item[1]) #Get source_number

        #bn_sample_vp_ch1=abcd['room_nos'][item[0]]['nsmix_f'][(vp-1)*2,:]
        noisy_mixture_all_channels_=np.zeros((max_no_of_channels_in_dataset, length_of_mixture))

        #channel_in_current_data=abcd['room_nos'][item[0]]['nsmix_f'].shape[1]

        noisy_mixture_all_channels_[:,:]=abcd['room_nos'][item[0]]['nsmix_f'][no_of_sources-1,:,:]  # number of sources in the mixture x number of channels present x length of signal

        #surface=anno_file['room_nos'][item[0]]['surface'][0]
        #volume=anno_file['room_nos'][item[0]]['volume'][0]
        #rt60=anno_file['room_nos'][item[0]]['rt60'][()].reshape(6)

        azimuth_spectrum=anno_file['room_nos'][item[0]]['azimuth_spectrum'][no_of_sources-1,:] #  3, 360    3: refers to 3 sources in the mixture

        azimuth=anno_file['room_nos'][item[0]]['azimuth'][:no_of_sources]



        #dimen=np.array(self.root_labels[item[0]]['dimension'])
        #diffusion=anno_file['room_nos'][item[0]]['diff']

        #sample={'bnsample':np.vstack([bn_sample_vp_ch1,bn_sample_vp_ch2]),'surface':surface,'volume':volume,'rt60':rt60,'room':int(item[0].split("_")[1]),'vp':vp}
        sample={'bnsample':noisy_mixture_all_channels_,'room':int(item[0].split("_")[1]),'no_of_sources':no_of_sources,'azimuth_spectrum':azimuth_spectrum,'azimuth':azimuth}

        return sample


'''
bn_dataset=binuaral_dataset('/home/psrivastava/source_localization/ICASSP/data_generation/train_random_ar_only_source_1_included.npy')

dataloader = DataLoader(bn_dataset, batch_size=4, shuffle=True, num_workers=0)



for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['bnsample'].size(),sample_batched['azimuth_spectrum'].size(),sample_batched['no_of_sources'].size(),sample_batched['azimuth'].size())

    if i_batch == 3:
        print(sample_batched['bnsample'].size())
        print(sample_batched['azimuth_spectrum'].size())
        #print(sample_batched['bnsample'][0,1,:].size())
        #print(stft(sample_batched['bnsample'][0,1,:].float()).shape)
        #print(sample_batched['volume'])
        #print(sample_batched['surface'])
        #print("rt60",sample_batched['rt60'])

        print(sample_batched["room"])

        plt.plot(np.linspace(0,180,num=360),sample_batched['azimuth_spectrum'][0,:])
        plt.savefig("batching_1.jpeg")

        plt.clf()

        plt.plot(np.linspace(0,180,num=360),sample_batched['azimuth_spectrum'][1,:])
        plt.savefig("batching_2.jpeg")

        plt.clf()

        plt.plot(np.linspace(0,180,num=360),sample_batched['azimuth_spectrum'][2,:])
        plt.savefig("batching_2.jpeg")


        sf.write("check.wav",sample_batched["bnsample"][1,0,:],16000)

        print(sample_batched['no_of_sources'])

        print(sample_batched['azimuth'])



        break
'''
'''
for x,j in enumerate(sample_batched['channel_in_current_data']):

    if j == 2 :
        print(j)
        print(sum(sample_batched['bnsample'][x,2,:]==0))
'''

## Codebase for the work : HOW TO (VIRTUALLY) TRAIN YOUR SOUND SOURCE LOCALIZER 
### Authors: Prerak Srivastava, Antoine Deleforge, Archontis Politis, Emmanuel Vincent.
### Paper :  https://hal.archives-ouvertes.fr/hal-03855912/document

### ./Data generation  

This directory contains scripts for the generation of room parameters, room impulse responses and speech mixtures.

**a)** **generate_simultor_params.py**. This script generates random simulation parameters for a desired number of rooms acording to some distribution.
Distributions corresponding to 3 real speaker localization datasets **DIRHA [1]**, **VOICEHOME2 [2]**, **STARSS22 [3]** are provided. The script depends on the params.yml file, from where the distribution of simulation parameters can be adjusted. The script generate 4 ".yml" files : 
* room_setup -> Dimensions, absorption coeffs, surface area, volume, 
* source_setup -> source position, Source directivity azimuth and elevation, 
* receiver_setup -> receiver position, receiver directivity azimuth and elevation, 
* noise_source_setup -> source position. File consisting of positions of extra source placed in the room setup, used particularly for generating speech mixtures with variable SNR's.

**b)** **receiver_definition_DIRHA.py**, **receiver_definition_STARSS22.py**, **receiver_definition_VOICEHOME2.py**. These scripts define the respective microphone arrays used in the real datasets. When called with required parameters these scripts return :
* individual position of each mic in the room,
* array barrycenter,
* azimuth and elevation of each individual mics and the center of the 2 mic array.

The distances between microphones are as follows :
* STARSS22 : 6.8CM  
* DIRHA : 30 CM
* VOICEHOME2 : 10.4 CM 

**c)** **generate_rirs.py**. This script generates room impulse responses on the basis of the yaml files generated by **a)**. It uses the following version of pyroomacoustics: https://github.com/LCAV/pyroomacoustics/tree/dev/dirpat.

**d)** **generate_noisy_mixture.py**. Creates noisy mixtures from the generated room impulse responses **c)**

The rest of the files in this directory are utility files.

### ./retreive_real_data 

This directory contains scripts that retreive real data if provided with correct directory of the respective real datasets.
* **get_DIRHA_data.py**
* **get_voicehome2_data.py**
* **get_starss22_data.py**

The output is saved in the .npz format, for easy access using numpy.

### ./train_scripts

Consists of all the scripts that are required for training the DOA system based on the generated simulated datasets. 
We used the DOA architecture presented in this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9357962

### ./srp_phat 

SRP-PHAT scripts used as a baseline. Can be used with simulated and real datasets (DIRHA [1], VOICEHOME2 [2] and STARSS22 [3]).

### ./test_scripts 

Consists of all the scripts that are used for testing the DOA estimation network on simulated and real datasets.

### References 

[1] M. Ravanelli, L. Cristoforetti, R. Gretter, M. Pellin, A. Sosi & M. Omologo (2015). The DIRHA-English corpus and related tasks for distant-speech recognition in domestic environments. In 2015 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU) (pp. 275-282). IEEE. https://arxiv.org/pdf/1710.02560.pdf

[2] N. Bertin, E. Camberlein, R. Lebarbenchon, E. Vincent, S. Sivasankaran, I. Illina & F. Bimbot (2019). VoiceHome-2, an extended corpus for multichannel speech processing in real homes. Speech Communication, 106, 68-78. https://hal.inria.fr/hal-01923108/document

[3] A. Politis, K. Shimada, P. Sudarsanam, S. Adavanne, D. Krause, Y. Koyama, & T. Virtanen (2022). STARSS22: A dataset of spatial recordings of real scenes with spatiotemporal annotations of sound events. arXiv preprint arXiv:2206.01948. https://arxiv.org/abs/2206.01948
 

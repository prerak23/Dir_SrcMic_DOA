#!/bin/bash
#OAR -q production
#OAR -p cluster='graffiti'
#OAR -l walltime=20
#OAR -O /home/psrivastava/logs_oarsub/oar_job.%jobid%.output
#OAR -E /home/psrivastava/logs_oarsub/oar_job.%jobid%.error
set -xv 

/home/psrivastava/source_localization/doa_estimation/dcase_mic_arr/train_scripts/D1_0000/run_cmd.sh





#!/bin/bash

#py=python3
py=python
exp=aistpp_fix
#seq_id="gLH_sBM_cAll_d16_mLH1_ch04"
#seq_id="gWA_sBM_cAll_d26_mWA1_ch10"
#seq_id="gWA_sFM_cAll_d27_mWA2_ch17"
#seq_id="gLO_sBM_cAll_d14_mLO4_ch05"
#seq_id="gHO_sFM_cAll_d19_mHO1_ch02"
#seq_id="gKR_sBM_cAll_d28_mKR0_ch09"
#seq_id="gKR_sBM_cAll_d30_mKR5_ch06"
#seq_id="gWA_sFM_cAll_d25_mWA3_ch04"
#seq_id="gWA_sBM_cAll_d26_mWA1_ch05"
#seq_id="gJB_sBM_cAll_d09_mJB5_ch10"
#seq_id="gLH_sFM_cAll_d16_mLH4_ch05"
#seq_id="gLO_sBM_cAll_d13_mLO3_ch06"
#seq_id="gKR_sBM_cAll_d28_mKR1_ch07"
#seq_id="gMH_sBM_cAll_d24_mMH2_ch08"
#seq_id="gJS_sBM_cAll_d03_mJS3_ch02"
#seq_id="gBR_sBM_cAll_d04_mBR2_ch01"
seq_id="gWA_sBM_cAll_d26_mWA1_ch05"
#seq_id="mambo"

$py inference/generate.py --data_dir=test_data --experiment_name=$exp \
    --seq_id $seq_id \
    --input_modalities="joint_angles_scaled,mel_ddcpca_scaled" \
    --output_modalities="joint_angles_scaled" \

$py analysis/aistplusplus_api/generate_video_from_mats.py --pred_mats_file generated/joint_angles_mats/${seq_id}.pkl_joint_angles_mats.generated.npy \
    --audio_file test_data/${seq_id}.mp3 \
    --trim_audio 2



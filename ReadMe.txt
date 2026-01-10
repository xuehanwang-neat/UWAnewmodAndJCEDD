To reproduce the simulation figures in this paper:
1.   Run Genmod/Gen_dataset.m in matlab to generate the dataset.
2.   Run Genmod/modem_learn_FullLMMSE_mse.py to train the neural network
3.   Run Genmod/modem_save_ZP.py to save the modulation matrix F as F.mat.
4.   After generating F.mat, copy F.mat to all other folders. This means the proposed modulation matrix.
5.   Run MSEfairness/main_MSEfairness.m to simulate Fig. 3 (Fairness NMSE vs SNR).
6.   Run perfectCSI/main_ber_fullLMMSE.m to simulate Fig. 4 (BER vs SNR under training parameters)
7.   Run perfectCSI/main_ber_fullLMMSE_robust.m to simulate Fig. 5 (BER vs SNR under different parameters)
8.   Run JCEDD_iter/main_JCEDD_iter.m to simulate Fig. 6 (BER vs iteration times)
9.   Run JCEDD_rho/main_JCEDD_rho.m to simulate Fig. 7 (BER vs \rho)

10.  Run JCEDD_SNR/main_JCEDD_SNR.m to simulate Fig. 8 (BER vs SNR with the joint receiver)

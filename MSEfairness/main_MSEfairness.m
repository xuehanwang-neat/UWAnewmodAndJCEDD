clear all ;
clc ;
rng(2025) ;
warning off ;
%% system parameters
fc = 12.5e3 ; % the carrier frequency (Hz)
roll_off = 0.65 ; % the rolling off factor of RC
Q = 4 ; % the truncated range for RC
B = 5e3 ; % the bandwidth
fs = B / (1+roll_off) ; % the sampling rate (Hz), equivalent to the bandwidth
T = 1 / fs ; % sampling period (s)
N = 256 ; % sequence length
Ng = 64 ; % guard length
Tg = Ng*T ; % guard interval
eye_N = eye(N) ;
F_OFDM = dftmtx(N) / sqrt(N) ;
F_OFDM = F_OFDM' ; % OFDM modulation matrix
load('F.mat','F') ;
F_propose = squeeze(F(1,:,:)+1j*F(2,:,:)) ;% proposed modulation matrix
F_SC = eye(N) ; % single carrier modulation matrix
M_ODDM = 8 ;
N_ODDM = N / M_ODDM ;
eye_ODDM = eye(M_ODDM) ;
dft_ODDM = dftmtx(N_ODDM) / sqrt(N_ODDM) ;
F_ODDM = kron(dft_ODDM',eye_ODDM) ; % ODDM modulation matrix
qam_mod = 2 ; % number of elements in QAM alphabets
qam_bit = log2(qam_mod) ; % bits per symbol for QPSK
num_bit = N * qam_bit ; % number of bits per transmission
% power per symbol
SNR_dB = 0:5:30 ;
SNR_linear = 10.^(SNR_dB/10) ;
sigma2_cand = 1 ./ SNR_linear / qam_bit * (2/(1+roll_off)) ;
length_SNR = length(SNR_dB) ;
mse_OFDM = zeros(length_SNR,1) ;
mse_SC = zeros(length_SNR,1) ;
mse_ODDM = zeros(length_SNR,1) ;
mse_propose = zeros(length_SNR,1) ;
%% Channel parameters
P = 8 ; % the number of channel paths
vmax = 10 ; % maximum mobility (kn)
c = 1500 ; % velocity of sound
amax = (vmax*1.852/3.6) / c ; % maximum doppler scaling factor
tau_max = 10e-3 ; % delay spread
lmax = tau_max / T ;
lmin = Q + floor(amax*(N+Ng-1)) ; % minimum normalized delay
lmax = lmax + lmin ; % maximum normalized delay 
decay_dB = 20 ; % the power difference from 0 to Tg
N_mc = 1e6 ; % the number of monte-carlo
for n_mc = 1:N_mc
    if mod(n_mc-1,N_mc/100) == 0
        fprintf('%3.2f%% finished \n',(n_mc-1)/(N_mc/100)) 
    end 
    % generate channel parameters
    [a_taps,l_taps,~,h_taps] = ...
        Gen_para(lmin,lmax,amax,P,decay_dB,Ng,fc,T) ;
    % channel matrix in the delay domain
    H = Gen_channel_mtx...
        (a_taps,l_taps,h_taps,P,fc,T,N,N+Ng,roll_off,Q) ;
    H_OFDM = H * F_OFDM ; % channel matrix for OFDM
    H_propose = H * F_propose ;
    H_SC = H * F_SC ;
    H_ODDM = H * F_ODDM ;
    for b = 1:length(sigma2_cand)
        sigma2 = sigma2_cand(b) ;
        % fully ICI-aware ZP-OFDM
        cov_OFDM = sigma2 * inv(H_OFDM'*H_OFDM+sigma2*eye_N) ;
        mse_OFDM(b) = mse_OFDM(b) ...
            + compute_NMSEfairness(diag(cov_OFDM)) ;
        % SC
        cov_SC = sigma2 * inv(H_SC'*H_SC+sigma2*eye_N) ;
        mse_SC(b) = mse_SC(b) ...
            + compute_NMSEfairness(diag(cov_SC)) ;
        % ODDM
        cov_ODDM = sigma2 * inv(H_ODDM'*H_ODDM+sigma2*eye_N) ;
        mse_ODDM(b) = mse_ODDM(b) ...
            + compute_NMSEfairness(diag(cov_ODDM)) ;
        % proposed modulation
        cov_propose = sigma2 * inv(H_propose'*H_propose+sigma2*eye_N) ;
        mse_propose(b) = mse_propose(b) ...
            + compute_NMSEfairness(diag(cov_propose)) ;
    end
end
mse_OFDM = mse_OFDM / N_mc  ;
mse_ODDM = mse_ODDM / N_mc  ;
mse_SC = mse_SC / N_mc  ;
mse_propose = mse_propose / N_mc ;
figure ;
plot(SNR_dB,mse_OFDM,'-+','Color',...
    [0.4940 0.1840 0.5560],'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,mse_SC,'-*','Color',...
    [0.8500 0.3250 0.0980],'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,mse_ODDM,'-^','Color',...
    [0.4660 0.6740 0.1880],'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,mse_propose,'m-p'...
    ,'LineWidth',1.5) ;
hold on ;
xlabel('E_{b}/N_{0} (dB)') ;
ylabel('Fiarness NMSE in (14)') ;
legend('OFDM','single carrier'...
    ,'ODDM','proposed modulation') ;
grid on ;
set(gca,'YScale','log') ;  
savefig('MSEfairness_train.fig') ;
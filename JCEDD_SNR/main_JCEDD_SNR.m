clear all ;
clc ;
rng(2025) ;
delete(gcp('nocreate')) ;
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
F_propose = double(F_propose) ;
F_SC = eye(N) ; % single carrier modulation matrix
M_ODDM = 8 ;
N_ODDM = N / M_ODDM ;
eye_ODDM = eye(M_ODDM) ;
dft_ODDM = dftmtx(N_ODDM) / sqrt(N_ODDM) ;
F_ODDM = kron(dft_ODDM',eye_ODDM) ; % ODDM modulation matrix
qam_mod = 2 ; % number of elements in QAM alphabets
qam_bit = log2(qam_mod) ; % bits per symbol for QPSK
num_bit = N * qam_bit ; % number of bits per transmission
sp = sqrt(1/2) * (randn(N,1)+1j*randn(N,1)) ; % pilot sequence
rho = 0.3 ; % pilot power
% SNR definition
SNR_dB = 0:5:30 ;
SNR_linear = 10.^(SNR_dB/10) ;
sigma2_cand = 1 ./ SNR_linear / qam_bit * (2/(1+roll_off)) ;
length_SNR = length(SNR_dB) ;
parpool(length_SNR) ;
ber_OFDM = zeros(length_SNR,1) ;
ber_SC = zeros(length_SNR,1) ;
ber_ODDM = zeros(length_SNR,1) ;
ber_propose = zeros(length_SNR,1) ;
ber_propose_perfect = zeros(length_SNR,1) ;
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
% for OMP channel estimation
P_min = 8 ;
P_max = 30 ; % number of virtual paths for reconstruction
Ma = 50 ; % number of virtual scaling grids
Mtau = 150 ; % number of virtual delay grids
% OMP dictionary matrix and virtual grids
[B_OMP_base,B_OMP_ini,lg,ag] = Gen_OMPbase_full...
    (sp,N,Ng,T,fc,roll_off,Q,Ma,amax,Mtau,lmin,lmax) ;
N_iter = 15 ; % maximum number of iterations
N_iter_in = 4 ;
N_iter_sum = N_iter*N_iter_in ;
N_mc = 1e4 ; % the number of monte-carlo
for n_mc = 1:N_mc
    if mod(n_mc-1,N_mc/100) == 0
        fprintf('%3.2f%% finished \n',(n_mc-1)/(N_mc/100)) 
    end 
    % generate channel parameters
    [a_taps,l_taps,~,h_taps] = ...
        Gen_para(lmin,lmax,amax,P,decay_dB,Ng,fc,T) ;
    H = Gen_channel_mtx...
        (a_taps,l_taps,h_taps,P,fc,T,N,N+Ng,roll_off,Q) ;
    H_propose = H * F_propose ;
    % generate data vector
    data_bit = randi([0 1],qam_bit,N) ;
    data_sym = qammod(data_bit,qam_mod,'InputType','bit','UnitAveragePower',true) ;
    data_sym = data_sym.' ;
    % sequence-level modulation
    s_tx_OFDM = sqrt(rho) * sp +...
        sqrt(1-rho) * F_OFDM * data_sym ; 
    s_tx_SC = sqrt(rho) * sp +...
        sqrt(1-rho) * F_SC * data_sym ;
    s_tx_ODDM = sqrt(rho) * sp +...
        sqrt(1-rho) * F_ODDM * data_sym ;
    s_tx_propose = sqrt(rho) * sp +...
        sqrt(1-rho) * F_propose * data_sym ; 
    % transmission
    r_rx_OFDM_wn = H * s_tx_OFDM ;
    r_rx_propose_wn = H * s_tx_propose ;
    r_rx_SC_wn = H * s_tx_SC ;
    r_rx_ODDM_wn = H * s_tx_ODDM ;
    n_norm = sqrt(1/2) * (randn(N+Ng,1)...
        + 1j*randn(size(N+Ng,1)));
    parfor b = 1:length(sigma2_cand)
        sigma2 = sigma2_cand(b) ;
        % additive noise
        r_rx_OFDM = r_rx_OFDM_wn + sqrt(sigma2) * n_norm ;
        r_rx_propose = r_rx_propose_wn + sqrt(sigma2) * n_norm ;
        r_rx_SC = r_rx_SC_wn + sqrt(sigma2) * n_norm ;
        r_rx_ODDM = r_rx_ODDM_wn + sqrt(sigma2) * n_norm ;
        % BER computation
        % OFDM
        [bit_est,~] = JCEDD(r_rx_OFDM,F_OFDM,...
            B_OMP_ini,B_OMP_base,sp,lg,ag,P_min,P_max,...
            fc,T,N,Ng,roll_off,Q,qam_mod,N_iter,N_iter_in,sigma2,rho) ;
        ber_OFDM(b) = ber_OFDM(b) ...
            + sum(sum(data_bit~=bit_est)) ;
        % SC
        [bit_est,~] = JCEDD(r_rx_SC,F_SC,...
            B_OMP_ini,B_OMP_base,sp,lg,ag,P_min,P_max,...
            fc,T,N,Ng,roll_off,Q,qam_mod,N_iter,N_iter_in,sigma2,rho) ;
        ber_SC(b) = ber_SC(b) ...
            + sum(sum(data_bit~=bit_est)) ;
        % ODDM
        [bit_est,~] = JCEDD(r_rx_ODDM,F_ODDM,...
            B_OMP_ini,B_OMP_base,sp,lg,ag,P_min,P_max,...
            fc,T,N,Ng,roll_off,Q,qam_mod,N_iter,N_iter_in,sigma2,rho) ;
        ber_ODDM(b) = ber_ODDM(b) ...
            + sum(sum(data_bit~=bit_est)) ;
        % proposed modulation
        [bit_est,~] = JCEDD(r_rx_propose,F_propose,...
            B_OMP_ini,B_OMP_base,sp,lg,ag,P_min,P_max,...
            fc,T,N,Ng,roll_off,Q,qam_mod,N_iter,N_iter_in,sigma2,rho) ;
        ber_propose(b) = ber_propose(b) ...
            + sum(sum(data_bit~=bit_est)) ;
        % proposed modulation with perfect CSI
        r_rx_data = r_rx_propose - sqrt(rho) * H * sp ;
        He = sqrt(1-rho) * H_propose ;
        data_rx = (He'*He+sigma2*eye_N) \ (He'*r_rx_data) ;
        bit_est = qamdemod(data_rx.',qam_mod,'OutputType','bit','UnitAveragePower',true) ;
        ber_propose_perfect(b) = ber_propose_perfect(b)...
            + sum(sum(data_bit~=bit_est)) ;
    end
end
ber_OFDM = ber_OFDM / N_mc / num_bit ;
ber_ODDM = ber_ODDM / N_mc / num_bit ;
ber_SC = ber_SC / N_mc / num_bit ;
ber_propose = ber_propose / N_mc / num_bit ;
ber_propose_perfect = ber_propose_perfect / N_mc / num_bit ;
figure ;
plot(SNR_dB,ber_OFDM,'-+','Color',...
    [0.4940 0.1840 0.5560],'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,ber_SC,'-*','Color',...
    [0.8500 0.3250 0.0980],'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,ber_ODDM,'-^','Color',...
    [0.4660 0.6740 0.1880],'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,ber_propose,'m-p'...
    ,'LineWidth',1.5) ;
hold on ;
plot(SNR_dB,ber_propose_perfect,'-v','Color',...
    [1.0000 0.4118 0.1608],'LineWidth',1.5) ;
hold on ;
xlabel('E_{b}/N_{0} (dB)') ;
ylabel('BER') ;
legend('OFDM','single carrier','ODDM',...
    'proposed modulation',...
    'proposed modulation (perfect CSI)') ;
grid on ;
set(gca,'YScale','log') ;  
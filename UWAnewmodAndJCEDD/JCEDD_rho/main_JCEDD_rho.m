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
load('F.mat','F') ;
F_propose = squeeze(F(1,:,:)+1j*F(2,:,:)) ;% proposed modulation matrix
F_propose = double(F_propose) ;
qam_mod = 2 ; % number of elements in QAM alphabets
qam_bit = log2(qam_mod) ; % bits per symbol for QPSK
num_bit = N * qam_bit ; % number of bits per transmission
SNR_dB = [30;25;20;15] ; 
SNR_linear = 10.^(SNR_dB/10) ;
sigma2_cand = 1 ./ SNR_linear / qam_bit * (2/(1+roll_off)) ;
sp = sqrt(1/2) * (randn(N,1)+1j*randn(N,1)) ; % pilot sequence
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
% Monto Carlo experiments
N_mc = 1e4 ;
rho_cand = (0.2:0.1:0.9).' ;
length_rho = length(rho_cand) ;
ber_30dB = zeros(length_rho,1) ;
ber_25dB = zeros(length_rho,1) ;
ber_20dB = zeros(length_rho,1) ;
ber_15dB = zeros(length_rho,1) ;
parpool(length_rho) ;
for n_mc = 1:N_mc
    if mod(n_mc-1,N_mc/100) == 0
        fprintf('%3.2f%% finished \n',(n_mc-1)/(N_mc/100)) 
    end
    % generate channel parameters and matrix
    [a_taps,l_taps,~,h_taps] = ...
        Gen_para(lmin,lmax,amax,P,decay_dB,Ng,fc,T) ;
    H = Gen_channel_mtx...
        (a_taps,l_taps,h_taps,P,fc,T,N,N+Ng,roll_off,Q) ;
    % generate data vector
    data_bit = randi([0 1],qam_bit,N) ;
    x = qammod(data_bit,qam_mod,'InputType','bit','UnitAveragePower',true) ;
    x = x.' ;
    n_norm = sqrt(1/2) * (randn(N+Ng,1)...
        + 1j*randn(size(N+Ng,1))) ;
    parfor b = 1:length_rho
        rho = rho_cand(b) ;
        % sequence-level modulation
        s_tx_propose = sqrt(rho)*sp + ...
            sqrt(1-rho) * F_propose * x ;
        % transmission
        r_rx_propose_wn = H * s_tx_propose ;
        % Eb/N0 = 30dB
        sigma2 = sigma2_cand(1) ;
        r_rx_propose = r_rx_propose_wn + sqrt(sigma2) * n_norm ;
        [bit_est,~] = JCEDD(r_rx_propose,F_propose,...
            B_OMP_ini,B_OMP_base,sp,lg,ag,P_min,P_max,...
            fc,T,N,Ng,roll_off,Q,qam_mod,N_iter,N_iter_in,sigma2,rho) ;
        ber_30dB(b) = ber_30dB(b) + ...
            sum(sum(abs(bit_est-data_bit))) ;
        % Eb/N0 = 25dB
        sigma2 = sigma2_cand(2) ;
        r_rx_propose = r_rx_propose_wn + sqrt(sigma2) * n_norm ;
        [bit_est,~] = JCEDD(r_rx_propose,F_propose,...
            B_OMP_ini,B_OMP_base,sp,lg,ag,P_min,P_max,...
            fc,T,N,Ng,roll_off,Q,qam_mod,N_iter,N_iter_in,sigma2,rho) ;
        ber_25dB(b) = ber_25dB(b) + ...
            sum(sum(abs(bit_est-data_bit))) ;
        % Eb/N0 = 20dB
        sigma2 = sigma2_cand(3) ;
        r_rx_propose = r_rx_propose_wn + sqrt(sigma2) * n_norm ;
        [bit_est,~] = JCEDD(r_rx_propose,F_propose,...
            B_OMP_ini,B_OMP_base,sp,lg,ag,P_min,P_max,...
            fc,T,N,Ng,roll_off,Q,qam_mod,N_iter,N_iter_in,sigma2,rho) ;
        ber_20dB(b) = ber_20dB(b) + ...
            sum(sum(abs(bit_est-data_bit))) ;
        % Eb/N0 = 15dB
        sigma2 = sigma2_cand(4) ;
        r_rx_propose = r_rx_propose_wn + sqrt(sigma2) * n_norm ;
        [bit_est,~] = JCEDD(r_rx_propose,F_propose,...
            B_OMP_ini,B_OMP_base,sp,lg,ag,P_min,P_max,...
            fc,T,N,Ng,roll_off,Q,qam_mod,N_iter,N_iter_in,sigma2,rho) ;
        ber_15dB(b) = ber_15dB(b) + ...
            sum(sum(abs(bit_est-data_bit))) ;
    end
end
ber_30dB_rho = ber_30dB / N_mc / num_bit ;
ber_25dB_rho = ber_25dB / N_mc / num_bit ;
ber_20dB_rho = ber_20dB / N_mc / num_bit ;
ber_15dB_rho = ber_15dB / N_mc / num_bit ;
figure ;
plot(rho_cand,ber_15dB_rho,'-+','Color',...
    [0.4940 0.1840 0.5560],'LineWidth',1.5) ;
hold on ;
plot(rho_cand,ber_20dB_rho,'-*','Color',...
    [0.8500 0.3250 0.0980],'LineWidth',1.5) ;
hold on ;
plot(rho_cand,ber_25dB_rho,'-^','Color',...
    [0.4660 0.6740 0.1880],'LineWidth',1.5) ;
hold on ;
plot(rho_cand,ber_30dB_rho,'m-p'...
    ,'LineWidth',1.5) ;
hold on ;
xlabel('\rho') ;
ylabel('BER') ;
legend('E_b/N_0=15 dB','E_b/N_0=20 dB'...
    ,'E_b/N_0=25 dB','E_b/N_0=30 dB') ;
grid on ;
set(gca,'YScale','log') ; 

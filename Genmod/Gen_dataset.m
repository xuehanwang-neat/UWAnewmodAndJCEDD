clear all ;
clc ;
N_train = 2e4 ;
N_test = 1e3 ;
fc = 12.5e3 ; % the carrier frequency (Hz)
roll_off = 0.65 ; % the rolling off factor of RRC
Q = 4 ; % the truncated range for RC
B = 5e3 ; % the bandwidth
fs = B / (1+roll_off) ; % the sampling rate (Hz), equivalent to the bandwidth
T = 1 / fs ; % sampling period (s)
N = 256 ; % sequence length
Ng = 64 ; % guard length
Tg = Ng*T ; % guard interval
P = 8 ; % the number of paths
vmax = 10 ; % maximum mobility (kn)
c = 1500 ; % velocity of sound
amax = (vmax*1.852/3.6) / c ; % maximum doppler scaling factor
tau_max = 10e-3 ; % delay spread
lmax = tau_max / T ;
lmin = Q + floor(amax*(N+Ng-1)) ; % minimum normalized delay
lmax = lmax + lmin ; % maximum normalized delay 
decay_dB = 20 ; % the power difference from 0 to Tg
H_train = zeros(N_train,2,N+Ng,N) ;
H_test = zeros(N_test,2,N+Ng,N) ;
for n_mc = 1:N_train
    [a_taps,l_taps,~,h_taps] = ...
        Gen_para(lmin,lmax,amax,P,decay_dB,Ng,fc,T) ;
    H = Gen_channel_mtx...
        (a_taps,l_taps,h_taps,P,fc,T,N,N+Ng,roll_off,Q) ;
    H_train(n_mc,1,:,:) = real(H) ;
    H_train(n_mc,2,:,:) = imag(H) ;
end
save('H_train.mat','H_train') ;
for n_mc = 1:N_test
    [a_taps,l_taps,~,h_taps] = ...
        Gen_para(lmin,lmax,amax,P,decay_dB,Ng,fc,T) ;
    H = Gen_channel_mtx...
        (a_taps,l_taps,h_taps,P,fc,T,N,N+Ng,roll_off,Q) ;
    H_test(n_mc,1,:,:) = real(H) ;
    H_test(n_mc,2,:,:) = imag(H) ;
end
save('H_test.mat','H_test') ;
function [a_taps,l_taps,A_taps,h_taps] = ...
    Gen_para(lmin,lmax,amax,P,decay_dB,Ng,fc,T)
% Generate UWA channel parameters
%% input arguments
% tau_max: maximum time delay (s)
% v_max: maximum mobility velocity (kn)
% P: the number of paths
% decay_dB: the power difference from 0 to Tg
% Tg: ZP duration
% fc: carrier frequency
% T: Nyquist symbol period
%% generate a_taps - scaling factors
a_taps = amax * cos(2*pi*rand(P,1)) ;
%% generate l_taps - normalized delay
l_taps = lmin + (lmax-lmin) * sort(rand(P,1)) ;
%% generate A_taps - path amplitude
decay_coe = 10^(decay_dB/10) ;
power_taps = exp(-log(decay_coe)*l_taps/Ng) ;
power_taps = power_taps ./ sum(power_taps) ;
A_taps = raylrnd(sqrt(power_taps/2)) ;
h_taps = A_taps .* exp(-1j*2*pi*fc*l_taps*T) ;



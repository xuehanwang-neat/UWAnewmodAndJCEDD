function H = Gen_channel_mtx...
    (a_taps,l_taps,h_taps,P,fc,T,M_tx,M_rx,roll_off,Q)
% generate channel matrix H in the time domain
%% Input arguments
% a_taps: time scaling for each path
% tau_taps: delay time for each path (s)
% A_taps: path amplitude for each path
% P: the number of paths
% fc: the carrier frequency (Hz)
% T: the sampling interval (s) 
% M_tx: the number of sampling points at Tx
% M_rx: the number of sampling points at Rx (M_rx>M_tx)
% roll_off: rolling off factor for the RC
% Q: the truncated length of RC
H = zeros(M_rx,M_tx) ;
% M_g = M_rx - M_tx ;
% sampling index at Tx
m_tx = 0:1:M_tx-1 ;
% sampling index at Rx
m_rx = (0:1:M_rx-1).' ;
for p = 1:P
    % fetch parameters for p-th path
    hp = h_taps(p) ; 
    lp = l_taps(p) ; 
    ap = a_taps(p) ;
    Doppler_p = exp(1j*2*pi*ap*fc*T*m_rx) ;
    G_index = (1+ap)*m_rx - m_tx - lp ;
    G_pulse = Nyquist_rcos(G_index,roll_off,Q) ;
    H = H + hp * G_pulse .* Doppler_p  ;
end


function [B_OMP_base,B_OMP_ini,lg,ag] = Gen_OMPbase_full...
    (sp,N,Ng,T,fc,roll_off,Q,Ma,amax,Mtau,lmin,lmax)
% generate base matrix/grids for 1D OMP
% sp: training sequence
% N: the number of subcarriers
% Ng: the length of guard interval
% T: Nyquist sampling period
% fc: carrier frequency
% Ma: the number of grids for ap
% amax: maximum doppler scaling factor
% Mtau: the number of grids for lp
% (lmin,lmax): delay range
Mg = Ma*Mtau ; % the number of total virtual sampling grids
ra = 2*amax/(Ma-1) ; % virtual doppler scaling resolution
rtau = (lmax-lmin)/(Mtau-1) ;% virtual delay resolution
i_index = (0:1:Mg-1).' ;%virtual sampling indices
ma_index = mod(i_index,Ma) ;
mtau_index = floor(i_index/Ma) ;
lg = mtau_index * rtau + lmin ; % virtual delay grids
ag = ma_index * ra - amax ; % virtual scaling grids
ag_T = ag.' ;
lg_T = lg.' ;
n_rx_index = (0:1:N+Ng-1).' ;
B_OMP_ini = zeros(N+Ng,Mg) ;
B_Doppler = exp(1j*2*pi*fc*T*(ag_T.*n_rx_index)) ;
B_OMP_base = zeros(N+Ng,Mg,N) ;
for n = 0:N-1
    sp_n = sp(n+1) ;
    l_sample = (1+ag_T).*n_rx_index-n-lg_T ;
    B_pulse = Nyquist_rcos(l_sample,roll_off,Q) ;
    B_OMP_ini = B_OMP_ini + sp_n*(B_Doppler.*B_pulse) ;
    B_OMP_base(1:end,1:end,n+1) = B_Doppler.*B_pulse ;
end
B_OMP_base = reshape(B_OMP_base,(N+Ng)*Mg,N) ;
B_OMP_base = sparse(B_OMP_base) ;
B_OMP_ini = sparse(B_OMP_ini) ;




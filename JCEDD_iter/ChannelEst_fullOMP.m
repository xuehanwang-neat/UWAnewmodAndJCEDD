function [a_taps,l_taps,h_taps,H_est] = ChannelEst_fullOMP...
    (r_CE,B_OMP,lg,ag,P_est,fc,T,N,Ng,roll_off,Q)
% Channel estimation via 1D OMP
a_taps = zeros(P_est,1) ;
l_taps = zeros(P_est,1) ;
B_est = zeros(N+Ng,P_est) ;
r_res = r_CE ;
for p = 1:P_est
    Phi_cor = abs(B_OMP' * r_res) ;
    [~,i] = max(Phi_cor) ;
    a_taps(p) = ag(i) ;
    ag(i) = [] ;
    l_taps(p) = lg(i) ;
    lg(i) = [] ;
    B_est(:,p) = B_OMP(:,i) ;
    B_OMP(:,i) = [] ;
    B_est_now = B_est(:,1:p) ;
    h_taps = (B_est_now'*B_est_now) \ (B_est_now'*r_CE) ;
    r_res = r_CE - B_est_now*h_taps ;
end
H_est = Gen_channel_mtx(a_taps,l_taps,h_taps,P_est,...
    fc,T,N,N+Ng,roll_off,Q) ;

    


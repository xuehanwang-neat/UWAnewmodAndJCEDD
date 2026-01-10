function [bit_est,H_est] = JCEDD(r,F,...
    B_OMP_ini,B_OMP_base,sp,lg,ag,P_min,P_max,fc,T,N,Ng,roll_off,Q,qam_mod,N_iter,N_iter_in,sigma2,rho)
% Joint channel estimation and data detection frwmework
% r: the received sequence
% F: modulation matrix
% B_OMP: OMP dictionary matrix
% sp: pilot sequence
% lg/ag: on-grid delay and doppler scaling
% P_max: maximum virtual paths
% fc: carrier frequency
% T: sampling period
% N: the number of subcarriers
% Ng: the length of guard interval
% roll_off: roll off factor for Nyquist pulse
% Q: truncated window length
% qam_mod: qam order
% rho: the power allocated to data symbols
% sigma2: noise variance
%% Initialize
Mg = length(ag) ;
B_OMP = sqrt(rho)*B_OMP_ini ;
eye_N = eye(N) ;
sp_pre = sqrt(rho) * sp ;
sigma2_est = 5*sigma2 ;
d_min = 1e8 ;
for n_iter = 1:N_iter
    dampling_pilot = 0.4 + 0.4 * (n_iter-1) / (N_iter-1) ;
    dampling_sigma2 = 0.4 + 0.4 * (n_iter-1) / (N_iter-1) ;
    P_now = P_min + (P_max-P_min) * (n_iter-1) / (N_iter-1) ;
    P_now = round(P_now) ;
    [~,~,~,H_est] = ChannelEst_fullOMP...
        (r,B_OMP,lg,ag,P_now,fc,T,N,Ng,roll_off,Q) ;
    r_DD = r - sqrt(rho) * H_est * sp  ;
    H_eq = sparse(sqrt(1-rho) * H_est * F) ;
    for n_iter_in = 1:N_iter_in
        x_est = (H_eq'*H_eq+sigma2_est*eye_N) ...
            \ (H_eq'*r_DD) ;
        bit_est = qamdemod(x_est.',qam_mod,...
            'OutputType','bit','UnitAveragePower',true) ;
        x_est = qammod(bit_est,qam_mod,...
            'InputType','bit','UnitAveragePower',true) ;
        x_est = x_est.' ;
        w_est = r_DD - H_eq * x_est ;
        norm_w = norm(w_est) ;
        d_current = norm_w * norm_w ;
        sigma2_est = dampling_sigma2 * d_current / N ...
            + (1-dampling_sigma2)*sigma2_est ;
        sigma2_est = max(sigma2_est,sigma2) ;
        if d_current < d_min
            H_est_final = H_est ;
            bit_est_final = bit_est ;
            d_min = d_current ;
        end
    end
    if n_iter == N_iter
        break ;
    else
        sp_new = sqrt(rho) * sp ...
            + sqrt(1-rho)*(F*x_est) ;
        sp_new = dampling_pilot * sp_new ...
            + (1-dampling_pilot) * sp_pre ;
        B_OMP = sparse(reshape(B_OMP_base*sp_new,N+Ng,Mg)) ;
        sp_pre = sp_new ;
    end
end
H_est = H_est_final ;
bit_est = bit_est_final ;
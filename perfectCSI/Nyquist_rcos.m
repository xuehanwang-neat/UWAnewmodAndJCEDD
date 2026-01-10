function a_sample = Nyquist_rcos(l_sample,roll_off,Q)
% compute the value of Nyquist filter 
% at off-grid sampling time
%% Input 
% l_sample - normalized sampling time
% rho - the raised factor
%% Output
% a_sample - values at the sampling time
sinc_l = sinc(l_sample) ;
cos_l = cos(roll_off*pi*l_sample) ./ ...
    (1-4*roll_off*roll_off*l_sample.*l_sample) ;
window_l = abs(l_sample)<Q ;
a_sample = sinc_l .* cos_l .* window_l ;


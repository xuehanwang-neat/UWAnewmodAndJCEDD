function nmse_result = compute_NMSEfairness(eh)
N = length(eh) ;
eh_opt = sum(eh) / N * ones(N,1) ;
nmse_result_norm = norm(eh_opt-eh,2) / norm(eh_opt,2) ;
nmse_result = nmse_result_norm*nmse_result_norm ;


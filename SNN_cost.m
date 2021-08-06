%% Evaluation of Cost
function [J, p_k, acc] = SNN_cost(Vv,n_ex,label) 
    %n_ex = number of examples
    %T_ex = time window of dynamic event
    %Vv = voltage spike train in output layer

    Y_k = zeros(size(Vv)); %y_k = diag(ones(n_ex,1)); %Array of labels
    T_ex = floor(size(Vv,2)/n_ex);
    y_k = zeros(size(Vv,1),n_ex);
%     fwin = [exp(-[floor(0.5*T_ex)-1:-1:0]/(0.5*T_ex)) ones(1,ceil(0.5*T_ex))]; % Weight function over time window
    fwin = [ones(1,1*T_ex)];
    H_k = Vv./(sum(Vv,1)+eps);
    for ii=1:n_ex
        y_k(label(ii),ii) = 1;
%         Y_k(label(ii),(ii-1)*T_ex+1 : ii*T_ex) = 1*fwin;
        H_k(:,(ii-1)*T_ex+1 : ii*T_ex) = H_k(:,(ii-1)*T_ex+1 : ii*T_ex) .*fwin;
        p_k(:,ii) = sum(H_k(:,(ii-1)*T_ex+1 : ii*T_ex),2)/sum(fwin);
    end

    % J = 1/(n_ex*T_ex) *sum(sum(-Y_k.*log(H_k +eps) -(1-Y_k).*log(1-H_k +eps), 1),2);
    J = 1/n_ex *sum(sum(-y_k.*log(p_k +eps) -(1-y_k).*log(1-p_k +eps), 1),2);
    acc = sum((sum(p_k.*y_k)))/n_ex;
end

    
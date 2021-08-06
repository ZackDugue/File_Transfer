
function [acc_train, acc_test, p_kSave_train, p_kSave_test, RT_train, RT_test] = discriminator(X_train, labels_all_train, label_train, columns_interest, folderSave, fname_Actsave, fname_ActTest_save, save_freq, to_test)

%% CLASSIFY from MULTI-column network

% load(dataset)
% X_input = data_all; 

accVal1 = [];
accVal2 = [];

acc_train = {};
acc_test = {};
p_kSave_train = {};
p_kSave_test = {};
RT_train = {};
RT_test = {};

rng(1);

% LOAD HYPERPARAMETERS OF NETWORK
optName = strcat(folderSave, '/autoenc_opt.mat');
opt = {};
load(optName)   
% L1Pos = []; L2Pos = []; L3Pos = [];
if numel(columns_interest) == 1
    tot_epochs = opt.tot_epochs(columns_interest(1));
else
    tot_epochs = min(opt.tot_epochs);
end

% ONCE
epoch = save_freq; 
prevFire = [];
for col2 = 1:numel(columns_interest)
    col = columns_interest(col2);
    fname_act_save2 = strcat(fname_Actsave, sprintf('column=%d_ep=%d.mat',col,epoch));
    %fname_act = sprintf('Act_column=%d.mat',col);
    load(fname_act_save2, 'L2H')
    prevFire = [prevFire; L2H];
end   

if to_test == true
    num_trainTest = 2;
else
    num_trainTest = 1;
end


for epoch2 = save_freq:save_freq:tot_epochs

    %% TOP (discriminator) Layer of the neural network
    
    topLayer = {};
    
    num_classes = length(unique(labels_all_train{1}));
    num_classes
    topLayer.dim = [1, num_classes];
    topLayer.numNeurons = num_classes;

    topLayer.D = sparse(zeros(topLayer.numNeurons));
    topLayer.S_d = sparse(diag(diag(ones(topLayer.numNeurons))));
    topLayer.S = sparse(zeros(topLayer.numNeurons));
    [X,Y] = meshgrid([0.5:1:topLayer.dim(2)-0.5],[topLayer.dim(1)-0.5:-1:0.5]);
    topLayer.nx = [ X(:) Y(:) ];

    topLayer.fire = [];
    topLayer.eta_learn = 1;
    topLayer.thresh = sparse(zeros(topLayer.numNeurons,1));

    topLayer.th = ones(topLayer.numNeurons,1);        %variable retina thresh
    topLayer.v_reset = 0 + 0.1*randn(topLayer.numNeurons,1).^2;     %Noise on activity field
    topLayer.v = sparse(zeros(topLayer.numNeurons,1)); %I.C. of v
    topLayer.H = sparse(zeros(topLayer.numNeurons,1)); % equivalent to "spikeMat"
    topLayer.y = sparse(0*ones(topLayer.numNeurons,1)); %I.C. of v
    topLayer.x = sparse(0*ones(topLayer.numNeurons,1)); %I.C. of v
    topLayer.ff = sparse(0*ones(topLayer.numNeurons,1)); %I.C. of v
    topLayer.fb = sparse(0*ones(topLayer.numNeurons,1)); %I.C. of v

    topLayer.Hwin = sparse(zeros(topLayer.numNeurons,250)); %ARBIT nTwin VALUE
    topLayer.Hint = sparse(zeros(topLayer.numNeurons,1));
    topLayer.Herr = sparse(zeros(topLayer.numNeurons,1));
    topLayer.Hdes = sparse(zeros(topLayer.numNeurons,1));

    topLayer.tau_v = 1;%normrnd(1,0.2,[selfOrgNet(num_layer).numNeurons,1]);%1*ones(selfOrgNet(num_layer).numNeurons,1);
    topLayer.tau_th = 1;%normrnd(1,0.2,[selfOrgNet(num_layer).numNeurons,1]);%1*ones(selfOrgNet(num_layer).numNeurons,1);
    topLayer.th_plus = 1;%normrnd(1,0.2,[selfOrgNet(num_layer).numNeurons,1]);%1*ones(selfOrgNet(num_layer).numNeurons,1);
    topLayer.v_th = 1;

    mu_W = 1;
    sigma_W = 0.5;

    W = normrnd(mu_W, sigma_W, [size(prevFire,1), topLayer.numNeurons]);
    topLayer.W = W./mean(W)*mu_W;    % Weight Matrix between Retina-LGN1 normrnd initialized


    for rr = 1:num_trainTest
        
        if rr == 1
            
            disp('learning run')
            prevFire = [];
            for col2 = 1:numel(columns_interest)
                col = columns_interest(col2);
                fname_act_save2 = strcat(fname_Actsave, sprintf('column=%d_ep=%d.mat',col,epoch));
                if ~isfile(fname_act_save2)
                    continue;
                else
                    load(fname_act_save2, 'L2H')                    
                    prevFire = [prevFire; L2H];
                end
                %fname_act = sprintf('Act_column=%d.mat',col);
                %load(fname_act_save2, 'L2H')
                %prevFire = [prevFire; L2H];
            end  
            max_epoch = 100;
            opts_train = [1,1,1];
            lr = 4;
            
            X_input = X_train{rr};
            labels_all = labels_all_train{rr};
            label = label_train{rr};
            label
            num_classes = length(unique(labels_all));
            n_ex = length(label);
            num_ex_eachClass = floor(size(X_input,2)/n_ex); 
            
        elseif rr == 2
            disp('test run')
            prevFire = [];
            for col2 = 1:numel(columns_interest)
                col = columns_interest(col2);
                fname_act_save2 = strcat(fname_ActTest_save, sprintf('column=%d_ep=%d.mat',col,epoch));
                if ~isfile(fname_act_save2)
                    continue;
                else
                    load(fname_act_save2, 'L2H')                    
                    prevFire = [prevFire; L2H];
                end
            end  
            max_epoch = 1;
            opts_train = [0,0,1];
            lr = 0;
            X_input = X_train{rr};
            labels_all = labels_all_train{rr};
            label = label_train{rr};
            label
            num_classes = length(unique(labels_all));
            n_ex = length(label);
            num_ex_eachClass = floor(size(X_input,2)/n_ex);     

        end 


        %% 
        % Time loop settings & Initiialization of LR optimization
        J_vec = []; lr_vec = []; conv_ctr = 0; minJ = 0.1; spdp = 1; %spdp=speedup
        %lr = 4;
        
        for ss2 = 1:max_epoch
            prevLayer = {};

            ss = ss2;
            actual_target = [];
            topLayer.eta_learn = lr;

            prevLayer.Hwin = sparse(zeros(size(prevFire,1),num_ex_eachClass*10));
            prevLayer.Hint = sparse(zeros(size(prevFire,1),1));

            Tend = size(X_input,2); dt = 0.1; t_intrvls = 0:Tend/dt;  
            tx = 0; Trot = num_ex_eachClass;

            Vv = zeros(topLayer.numNeurons, t_intrvls(end));
            Vh = zeros(topLayer.numNeurons, t_intrvls(end));

        %     Vv(:,1) = topLayer.v;
        %     Vh(:,1) = topLayer.H;

            topLayer.v = zeros(topLayer.numNeurons,1);
            topLayer.H = sparse(zeros(topLayer.numNeurons,1));
            topLayer.x = sparse(zeros(topLayer.numNeurons,1));
            topLayer.y = sparse(zeros(topLayer.numNeurons,1));
            topLayer.Hwin = sparse(zeros(topLayer.numNeurons,250));
            topLayer.Hint = sparse(zeros(topLayer.numNeurons,1));
            topLayer.Hdot = sparse(zeros(topLayer.numNeurons,1));
            topLayer.Herr = sparse(zeros(topLayer.numNeurons,1));
            topLayer.thresh = zeros(topLayer.numNeurons,1);
            topLayer.th = zeros(topLayer.numNeurons,1);


            % Time loop
            STDP_topLayer = logical(opts_train(1)); STDP_dt = 5*dt;
            opti_lr     = logical(opts_train(2));
            state_reset = logical(opts_train(3));
            
            debugMode2 = logical(0);
            debugMode = logical(0);

            tx = 0; ts = 1;
            Tend
            for tt = 0:dt:Tend-dt

                if mod(tt,1) == 0
                    tx = tx + 1;
                    topLayer.Hdes = zeros(num_classes,1);
%                     disp('labels_all')
%                     labels_all(tx)
                    topLayer.Hdes(labels_all(tx)) = 1; 
                    
                    if mod(tt,Trot)==0
                        if state_reset == true
                            
                            prevLayer.Hwin = sparse(zeros(size(prevFire,1),num_ex_eachClass*10));
                            prevLayer.Hint = sparse(zeros(size(prevFire,1),1));
            
                            topLayer.v = zeros(topLayer.numNeurons,1);
                            topLayer.H = sparse(zeros(topLayer.numNeurons,1));
                            topLayer.x = sparse(zeros(topLayer.numNeurons,1));
                            topLayer.y = sparse(zeros(topLayer.numNeurons,1));
                            topLayer.Hwin = sparse(zeros(topLayer.numNeurons,250));
                            topLayer.Hint = sparse(zeros(topLayer.numNeurons,1));
                            topLayer.th = zeros(topLayer.numNeurons,1);
                        end
                    end      
                end
                
                %ts, size(prevFire)
                prevLayer.Hwin = circshift(prevLayer.Hwin,1,2); 
                prevLayer.Hwin(:,1) = prevFire(:,ts); 
                prevLayer.Hint = sum(prevLayer.Hwin, 2);   % Top Hat filter  

                if range(prevLayer.Hint) == 0
        %                     num_layer, range(selfOrgNet(num_layer-1).Hint)
                else
                    prevLayer.Hint = prevLayer.Hint/range(prevLayer.Hint);
                end

                topLayer.x = max( topLayer.W' * prevLayer.Hint - topLayer.thresh, 0);  
                [wink, maxInd] = maxk(topLayer.x,1); % find Index of k best performers, a.k.a. Winners    
                topLayer.x(topLayer.x<wink) = 0;

                % topLayer.x(maxInd) = topLayer.x(maxInd) .* (5*topLayer.v_th)/(wink(1)+eps);
                topLayer.x(maxInd) = topLayer.x(maxInd) .* (5*8)/(wink(1)+eps);
                topLayer.x = sparse(topLayer.x);
                topLayer.fb = 1*topLayer.S*topLayer.H;
                topLayer.ff = topLayer.x;
                topLayer.v = RK4(@(v)(-1./topLayer.tau_v .*v + topLayer.fb + topLayer.ff),  dt,topLayer.v);
                topLayer.th = RK4(@(th)(1./topLayer.tau_th.*(topLayer.v_th-th).*(1-topLayer.H) + topLayer.th_plus.*topLayer.H),  dt, topLayer.th);

                if debugMode
                    if mod(tt,dt) == 0
                        figure(5),
                        cla()
                        plot(topLayer.v)
                        hold on
                        plot(topLayer.th)
                        plot(topLayer.x)
                        hold off
                        sprintf('# of firing neurons:='), length(find(topLayer.v >= topLayer.th))
                        figure(6)
                        tmp = topLayer.W'* prevLayer.Hint; 
                        hist(tmp)
                    end
                end        

                topLayer.fire = find(topLayer.v >= topLayer.th);
                topLayer.v(topLayer.fire) = topLayer.v_reset(topLayer.fire);
                topLayer.H = sparse(zeros(topLayer.numNeurons,1));
                topLayer.H(topLayer.fire) = ones(length(topLayer.fire),1);

                if (STDP_topLayer == true && mod(tt,STDP_dt) == 0)    % Solve W2 Matrix dynamical system & Update thresh

                    topLayer.Herr = topLayer.H - topLayer.Hdes;
                    tmp = (prevLayer.Hint*(-1*topLayer.Herr'));

                    if max(tmp(:))~=0
                        topLayer.W = topLayer.W + STDP_dt*(topLayer.eta_learn* tmp)/max(tmp(:));
                    end

                    topLayer.W = 5*(topLayer.W)./range(topLayer.W);

                    if debugMode2
                        disp('inside')
                        figure(4)
                        title('after update')
                        imagesc(topLayer.W')
                        colorbar
        %                 for num_layer = 2:total_layers
        %                     subplot(1,total_layers-1,num_layer-1)
        %                     imagesc(topLayer.W')
        %                     colorbar
        %                 end
                    end            
                end

        %         if debugMode2
        %             tt, lr
        %             if mod(tt,1) == 0
        %                 figure(2),
        %                 subplot(1,3,1)
        %                 axis ij image,
        %                 scatter(L1Pos(:,1),L1Pos(:,2),'k','filled')
        %                 hold on 
        %                 fireN = find(X_input(:,floor(tt)+1) == 1);
        %                 scatter(L1Pos(fireN,1),L1Pos(fireN,2),'r','filled')
        % 
        %                 subplot(1,3,2)
        %                 axis ij image,
        %                 scatter(L2Pos(:,1),L2Pos(:,2),'k','filled')
        %                 hold on
        %                 fireN = find(L2H(:,int8(tt/0.1+1)) == 1);
        %                 scatter(L2Pos(fireN,1),L2Pos(fireN,2),'r','filled')
        % 
        %                 subplot(1,3,3),
        %                 axis ij image,
        %                 scatter(topLayer.nx(:,1),topLayer.nx(:,2),'k','filled')
        %                 hold on
        %                 scatter(topLayer.nx(topLayer.fire,1),topLayer.nx(topLayer.fire,2),'r','filled')
        % 
        %                 pause(0.2)
        %             end
        %         end

                if mod(tt,dt) == 0

                    Vv(:,ts) = topLayer.v;
                    Vh(:,ts) = topLayer.H;
                    actual_target(end+1) = labels_all(tx);
                    ts = ts + 1;
                end
            end

            %[J, p_k, acc] = SNN_cost2(Vv,n_ex,label,actual_target);

            n_ex = length(label);
            n_ex, label
            [J, p_k, acc] = SNN_cost(Vv, n_ex, label);
            J, p_k

            tmp_labels = label;
            [classAcc, classInd] = max(p_k,[],1);
            %accVec = [length(find(abs(classInd - tmp_labels)==0)), mean(diag(p_k))];

            %[J, p_k, acc] = SNN_cost2(Vv,n_ex,label,actual_target);
            % J, p_k, acc

            J_vec = [J_vec J];
            lr_vec = [lr_vec lr];

            %Optimization of learning rate
            if lr == 0
                
                accVal1 = [accVal1, acc];
                accVal2 = [accVal2, length(find(abs(classInd - tmp_labels)==0))];
                accVal = [acc*100, length(find(abs(classInd - tmp_labels)==0))/length(label)*100];
                if rr == 1
                    acc_train{end+1} = accVal;
                    p_kSave_train{end+1} = p_k;
                    RT_train{end+1} = Vv;
                elseif rr == 2
                    acc_test{end+1} = accVal;
                    p_kSave_test{end+1} = p_k;
                    RT_test{end+1} = Vv;
                end
                
                break
            end


            
            %Optimization of learning rate
            if opti_lr
                if lr > 0.5/Tend
                    if length(J_vec)==1 || J_vec(end) < J_vec(end-1) * 0.9 || J < minJ
                        lr = lr/2,
                    elseif J_vec(end) > J_vec(end-1) / 0.9
                        lr = lr*1.5,
                    else
                        conv_ctr = conv_ctr+1; lr,
                        if conv_ctr >= 20, disp('Loss converged'), 
                            %lr = 1/Tend ; 
                            lr = 0;
                        end 
                    end
                else
        %             sprintf('converged')
                    lr = 0
                    %break;
        %             return;
                end
            end
            
            if ss2 == max_epoch-1
                lr = 0;
            end

            sprintf('ss = %d',ss)

        end
    end
end








function [] = unsupervisedTrain2_test(X_input, total_columns, total_epochs, save_freq, folderSave, fname_AEsave)

%% DEFINING HYPERPARAMETERS OF THE NETWORK

opt = {};
opt.total_layers = 3;

opt.nTwin = [];
% nTwin_L1= normrnd(150, 50, [1,total_columns]);
% nTwin_L1(nTwin_L1<1) = 1;   
% nTwin_L1 = floor(nTwin_L1);
nTwin_L1 = ones(1,total_columns);
opt.nTwin = [opt.nTwin; nTwin_L1];

opt.numNeurons = [];
opt.numNeurons_L1 = 8080*ones(1,total_columns);
opt.numNeurons = [opt.numNeurons; opt.numNeurons_L1];

for ii = 2:opt.total_layers-1
    nTwin_col = normrnd(150, 50, [1,total_columns]);
    nTwin_col(nTwin_col<1) = 1;   
    nTwin_col = floor(nTwin_col);
    %nTwin_col(1:end) = 1;
    opt.nTwin = [opt.nTwin; nTwin_col];

    rndNumNeurons = floor(normrnd(200,50,[1, total_columns]));
    opt.numNeurons = [opt.numNeurons; rndNumNeurons];
end

% Final layer is the same as the first layer
opt.numNeurons = [opt.numNeurons; opt.numNeurons(1,:)];
opt.nTwin = [opt.nTwin; opt.nTwin(1,:)];

opt.tau_v = [ones(1, total_columns); 1*ones(1, total_columns); ones(1, total_columns); ones(1, total_columns); ones(1, total_columns)];
opt.tau_th = [ones(1, total_columns); 2*ones(1, total_columns); 1*ones(1, total_columns); ones(1, total_columns); ones(1, total_columns)];
opt.th_plus = [ones(1, total_columns); 2*ones(1, total_columns); 1*ones(1, total_columns); ones(1, total_columns); ones(1, total_columns)];
opt.v_th = [ones(1, total_columns); 5*ones(1, total_columns); 1*ones(1, total_columns); ones(1, total_columns);ones(1, total_columns)];

opt.kWTA = [ones(1,total_columns); 50*ones(1, total_columns); 50*ones(1, total_columns); 50*ones(1,total_columns); 50*ones(1,total_columns)];

tot_epochs = [];
% opt.numNeurons
% opt.nTwin

for colId = 3:3%total_columns
    
   
    %% CREATING/GROWING SPIKING NETWORK USING HYPERPARAMETERS

    selfOrgNet = {};
    
    Tend = size(X_input,2); dt = 0.1;
    t_intrvls = 0:Tend/dt;
    
    % CREATING LAYER-1 (INPUT SENSORY GEOMETRY)
    
    num_layer = 1;
    selfOrgNet(num_layer).dim = [80,101];
    selfOrgNet(num_layer).numNeurons = opt.numNeurons(num_layer,colId);
    [X,Y] = meshgrid([0.5:1:selfOrgNet(num_layer).dim(2)-0.5],[selfOrgNet(num_layer).dim(1)-0.5:-1:0.5]); 
    selfOrgNet(num_layer).nx = [ X(:) Y(:) ];

    selfOrgNet(num_layer).ri = 1.0;
    selfOrgNet(num_layer).ro = 1.0;

    selfOrgNet(num_layer).D = sparse(zeros(selfOrgNet(num_layer).numNeurons));
    selfOrgNet(num_layer).S_d = sparse(zeros(selfOrgNet(num_layer).numNeurons));
    selfOrgNet(num_layer).S = sparse(zeros(selfOrgNet(num_layer).numNeurons));
    selfOrgNet(num_layer).eta_learn = 1;

    selfOrgNet(num_layer).thresh = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
    selfOrgNet(num_layer).th = ones(selfOrgNet(num_layer).numNeurons,1);        %variable retina thresh
    selfOrgNet(num_layer).v_reset = 0 + 0.1*randn(selfOrgNet(num_layer).numNeurons,1).^2;     %Noise on activity field
    selfOrgNet(num_layer).v = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); %I.C. of v
    selfOrgNet(num_layer).H = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); % equivalent to "spikeMat"
    selfOrgNet(num_layer).x = sparse(0*ones(selfOrgNet(num_layer).numNeurons,1)); %I.C. of v
    selfOrgNet(num_layer).ff = sparse(0*ones(selfOrgNet(num_layer).numNeurons,1)); %I.C. of v
    selfOrgNet(num_layer).fb = sparse(0*ones(selfOrgNet(num_layer).numNeurons,1)); %I.C. of v

    selfOrgNet(num_layer).Hwin = sparse(zeros(selfOrgNet(num_layer).numNeurons,opt.nTwin(num_layer, colId)));
    selfOrgNet(num_layer).Hint = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
    selfOrgNet(num_layer).Herr = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
    selfOrgNet(num_layer).Hdes = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));

    selfOrgNet(num_layer).fire = [];
    selfOrgNet(num_layer).firedMat = cell(1,length(t_intrvls));

    selfOrgNet(num_layer).tau_v = 1; 
    selfOrgNet(num_layer).tau_th = 1; 
    selfOrgNet(num_layer).th_plus = 1; 
    selfOrgNet(num_layer).v_th = 1;

    selfOrgNet(num_layer).htmp = zeros(selfOrgNet(num_layer).numNeurons,1); % Heatmap # of times each neuron spikes
    selfOrgNet(num_layer).eta = []; 

    total_layers = opt.total_layers; %(Including input and output layer)
    
    neurons_layers = zeros(total_layers,1);
    for num_layer2 = 1:total_layers
        num_layer = num_layer2;
        neurons_layers(num_layer) = opt.numNeurons(num_layer, colId);
    end
    
    dim_layers = {};
    dim_layers(1).s = [80,101];
    dim_layers(2).s = [10,10];
    dim_layers(3).s = [80,101];
    
    
    % CREATING 2+ LAYERS
    
    for num_layer2 = 2:total_layers
        num_layer = num_layer2;
        selfOrgNet(num_layer).dim = dim_layers(num_layer).s;
        
        if num_layer ~= total_layers

            selfOrgNet(num_layer).numNeurons = neurons_layers(num_layer); 
            selfOrgNet(num_layer).ri = 1; 
            selfOrgNet(num_layer).ro = 2;
            
            selfOrgNet(num_layer).nx = 10*rand(selfOrgNet(num_layer).numNeurons,2);
            %selfOrgNet(num_layer).S_d = sparse(eye(selfOrgNet(num_layer).numNeurons));
            %selfOrgNet(num_layer).S = sparse(zeros(selfOrgNet(num_layer).numNeurons));

            % when we have an adjacency matrix in each layer
            selfOrgNet(num_layer).D = squareform(pdist(selfOrgNet(num_layer).nx));
            selfOrgNet(num_layer).S = 10*(selfOrgNet(num_layer).D < selfOrgNet(num_layer).ri)- 1.5*(selfOrgNet(num_layer).D > selfOrgNet(num_layer).ro).*exp(-selfOrgNet(num_layer).D / 10); 
            selfOrgNet(num_layer).S = selfOrgNet(num_layer).S - diag(diag(selfOrgNet(num_layer).S)); 
            selfOrgNet(num_layer).S_d = 1*(selfOrgNet(num_layer).D < selfOrgNet(num_layer).ri)- 0.15*(selfOrgNet(num_layer).D > selfOrgNet(num_layer).ro).*exp(-selfOrgNet(num_layer).D / 10); 

             
            selfOrgNet(num_layer).fire = [];
            selfOrgNet(num_layer).firedMat = cell(1,length(t_intrvls));
            selfOrgNet(num_layer).eta_learn = 1;

            selfOrgNet(num_layer).thresh = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));

            selfOrgNet(num_layer).th = ones(selfOrgNet(num_layer).numNeurons,1);        %variable retina thresh
            selfOrgNet(num_layer).v_reset = 0 + 0.1*randn(selfOrgNet(num_layer).numNeurons,1).^2;     %Noise on activity field
            selfOrgNet(num_layer).v = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); %I.C. of v
            selfOrgNet(num_layer).H = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); % equivalent to "spikeMat"
            selfOrgNet(num_layer).x = sparse(0*ones(selfOrgNet(num_layer).numNeurons,1)); %I.C. of v
            selfOrgNet(num_layer).ff = sparse(0*ones(selfOrgNet(num_layer).numNeurons,1)); %I.C. of v
            selfOrgNet(num_layer).fb = sparse(0*ones(selfOrgNet(num_layer).numNeurons,1)); %I.C. of v

            selfOrgNet(num_layer).Hwin = sparse(zeros(selfOrgNet(num_layer).numNeurons,opt.nTwin(num_layer, colId)));
            selfOrgNet(num_layer).Hint = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
            selfOrgNet(num_layer).Herr = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
            selfOrgNet(num_layer).Hdes = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
            
            selfOrgNet(num_layer).tau_v = opt.tau_v(num_layer, colId);
            selfOrgNet(num_layer).tau_th = opt.tau_th(num_layer, colId);
            selfOrgNet(num_layer).th_plus = opt.th_plus(num_layer, colId);
            selfOrgNet(num_layer).v_th = opt.v_th(num_layer, colId);
            
            mu_W = 1;
            sigma_W = 0.5;
            
            W = normrnd(mu_W, sigma_W, [selfOrgNet(num_layer-1).numNeurons,selfOrgNet(num_layer).numNeurons]);
            selfOrgNet(num_layer).W = W./mean(W)*mu_W;    % Weight Matrix between Retina-LGN1 normrnd initialized

        else
            
            selfOrgNet(num_layer).numNeurons = neurons_layers(num_layer); 
            selfOrgNet(num_layer).ri = 1.0; 
            selfOrgNet(num_layer).ro = 1.0;
            

            selfOrgNet(num_layer).nx = selfOrgNet(1).nx;
            selfOrgNet(num_layer).D = selfOrgNet(1).D;
            selfOrgNet(num_layer).S_d = selfOrgNet(1).S_d;
            selfOrgNet(num_layer).S = selfOrgNet(1).S;
            
            
            selfOrgNet(num_layer).fire = [];
            selfOrgNet(num_layer).firedMat = cell(1,length(t_intrvls));
            selfOrgNet(num_layer).eta_learn = 1;

            selfOrgNet(num_layer).thresh = selfOrgNet(1).thresh;

            selfOrgNet(num_layer).th = selfOrgNet(1).th;      %variable retina thresh
            selfOrgNet(num_layer).v_reset = selfOrgNet(1).v_reset;     %Noise on activity field
            selfOrgNet(num_layer).v = selfOrgNet(1).v; %I.C. of v
            selfOrgNet(num_layer).H = selfOrgNet(1).H; % equivalent to "spikeMat"
            selfOrgNet(num_layer).x = selfOrgNet(1).x; %I.C. of v
            selfOrgNet(num_layer).ff = selfOrgNet(1).ff; %I.C. of v
            selfOrgNet(num_layer).fb = selfOrgNet(1).fb; %I.C. of v

            selfOrgNet(num_layer).Hwin = selfOrgNet(1).Hwin;
            selfOrgNet(num_layer).Hint = selfOrgNet(1).Hint;
            selfOrgNet(num_layer).Herr = selfOrgNet(1).Herr;
            selfOrgNet(num_layer).Hdes = selfOrgNet(1).Hdes;
            
            selfOrgNet(num_layer).tau_v = selfOrgNet(1).tau_v;
            selfOrgNet(num_layer).tau_th = selfOrgNet(1).tau_th;
            selfOrgNet(num_layer).th_plus = selfOrgNet(1).th_plus;
            selfOrgNet(num_layer).v_th = selfOrgNet(1).v_th;
            
            mu_W = 1;
            sigma_W = 0.5;

            W = normrnd(mu_W, sigma_W, [selfOrgNet(num_layer-1).numNeurons,selfOrgNet(num_layer).numNeurons]);
            selfOrgNet(num_layer).W = W./mean(W)*mu_W;    % Weight Matrix between Retina-LGN1 normrnd initialized            

        end
        
    end
    
   
    %% RUN DATA THROUGH NETWORK
    
    %%%% Time loop settings & Initialization of LR optimization
    
    netLossVal = cell(1,total_epochs);
    J_vec = []; lr_vec = []; lr = 1; conv_ctr = 0; minJ = 0.1; spdp = 1; %spdp=speedup factor
    
    for ss = 44:44%total_epochs
        
        if mod(ss, save_freq) == 0
            fname_load = strcat(folderSave, sprintf('/AE/AE_column=%d_ep=%d.mat', colId, ss));

            if ~isfile(fname_load)
                continue;
            end
            
            load(fname_load, 'selfOrgNet')
            lr = 0;
        else
            continue;
        end
        
        for num_layer2 = 2:total_layers        
            num_layer = num_layer2;
            selfOrgNet(num_layer).eta_learn = lr;
        end
        
        Tend = size(X_input,2); dt = 0.1; t_intrvls = 0:Tend/dt;  tx = 0; Trot = 1*Tend; ts = 0;  
        
        % Re-Initialization of special quantities after every epoch
        for num_layer2 = 1:total_layers
            num_layer = num_layer2;
            
            selfOrgNet(num_layer).Hwin = sparse(zeros(selfOrgNet(num_layer).numNeurons,opt.nTwin(num_layer,colId)));
            selfOrgNet(num_layer).Hint = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
            selfOrgNet(num_layer).Hdot = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); 
            selfOrgNet(num_layer).Herr = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
            selfOrgNet(num_layer).thresh = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); 
            selfOrgNet(num_layer).v = 0*ones(selfOrgNet(num_layer).numNeurons,1);     
            selfOrgNet(num_layer).H = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); 
            selfOrgNet(num_layer).x = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); 
            selfOrgNet(num_layer).th = ones(selfOrgNet(num_layer).numNeurons,1);                 
        end
        
        % Time loop
        STDP        = logical(1); STDP_dt = 5*dt; 
        opti_lr     = logical(1);
        state_reset = logical(1);
        debugMode   = logical(0);
        debugMode2  = logical(0);

        L2v = zeros(selfOrgNet(2).numNeurons, Tend*10);
        L2H = zeros(selfOrgNet(2).numNeurons, Tend*10);
        L3v = zeros(selfOrgNet(3).numNeurons, Tend*10);
        L3H = zeros(selfOrgNet(3).numNeurons, Tend*10);
        
        lossVal = [];
        lossVal2 = [];
        tic
   
        for tt = 0:dt:Tend-dt
        
            if mod(tt,1) == 0

                tx = tx+1;
                Xtmp = X_input(:,tx); 
                %Input sensors begin firing
                num_layer = 1;
                selfOrgNet(num_layer).eta = sparse(Xtmp(:)); 
                % Discontinuous Update rule
                fireR = find(selfOrgNet(num_layer).eta >= selfOrgNet(num_layer).th); 
                selfOrgNet(num_layer).H = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
                selfOrgNet(num_layer).H(fireR,1) = ones(length(fireR),1);
                selfOrgNet(total_layers).Hdes = selfOrgNet(1,1).H;

                if mod(tt,Trot)==0
                    if state_reset == true
                        % Reset all parameters after every new object
                        for num_layer2 = 2:total_layers
                            num_layer = num_layer2;
                            selfOrgNet(num_layer).v = zeros(selfOrgNet(num_layer).numNeurons,1);
                            selfOrgNet(num_layer).H = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
                            selfOrgNet(num_layer).x = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
                            if num_layer ~= total_layers
                                selfOrgNet(num_layer).Hwin = sparse(zeros(selfOrgNet(num_layer).numNeurons,opt.nTwin(num_layer,colId))); 
                            else
                                selfOrgNet(num_layer).Hwin = sparse(zeros(selfOrgNet(num_layer).numNeurons,opt.nTwin(1,colId))); 
                            end
                            selfOrgNet(num_layer).Hint = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
                            selfOrgNet(num_layer).th = 0*ones(selfOrgNet(num_layer).numNeurons,1);
                        end
                    end
                end
            end
               
            % FEEDFORWARD NEURAL NET ACTIVITY
            
            for num_layer2 = 2:total_layers
                num_layer = num_layer2;
                
                selfOrgNet(num_layer-1).Hwin = circshift(selfOrgNet(num_layer-1).Hwin,1,2); 
                selfOrgNet(num_layer-1).Hwin(:,1) = selfOrgNet(num_layer-1).H; 
                selfOrgNet(num_layer-1).Hint = sum(selfOrgNet(num_layer-1).Hwin, 2);   % Top Hat filter  
                

                if range(selfOrgNet(num_layer-1).Hint) == 0
%                     num_layer, range(selfOrgNet(num_layer-1).Hint)
                else
                    selfOrgNet(num_layer-1).Hint = selfOrgNet(num_layer-1).Hint/range(selfOrgNet(num_layer-1).Hint);
                end
                
                % Inputs to LGN: Weights -> Activation function -> Competition rule
                if num_layer ~= total_layers
                    selfOrgNet(num_layer).x = max( selfOrgNet(num_layer).W' * selfOrgNet(num_layer-1).Hint - selfOrgNet(num_layer).thresh, 0); % amount by how much input signal lies over thresh % ReLU activation function
                else
                    selfOrgNet(num_layer).x = max( selfOrgNet(num_layer).W' * selfOrgNet(num_layer-1).Hint - selfOrgNet(num_layer).thresh, 0); 
                end


                if num_layer ~= total_layers
                    [wink, maxInd] = maxk(selfOrgNet(num_layer).x,opt.kWTA(num_layer,colId)); % find Index of k best performers, a.k.a. Winners
                else
                    
                    objSize = sum(selfOrgNet(total_layers).Hdes);
                    if objSize == 0
                        objSize = 1;
                    end
                    %[wink, maxInd] = maxk(selfOrgNet(num_layer).x,opt.kWTA(num_layer,colId)); % find Index of k best performers, a.k.a. Winners
                    [wink, maxInd] = maxk(selfOrgNet(num_layer).x,objSize); % find Index of k best performers, a.k.a. Winners
                    
                    %[wink, maxInd] = maxk(selfOrgNet(num_layer).x,selfOrgNet(num_layer).numNeurons                [wink, maxInd] = maxk(selfOrgNet(num_layer).x,100); % find Index of k best performers, a.k.a. Winners); % find Index of k best performers, a.k.a. Winners
                end   

                % LGN Competition rule
                selfOrgNet(num_layer).x(selfOrgNet(num_layer).x<wink(end)) = 0;     % allow k best performers to participate

                if num_layer ~= total_layers
                    selfOrgNet(num_layer).x(maxInd) = (selfOrgNet(num_layer).x(maxInd)/(wink(1)+eps)).^8 .*selfOrgNet(num_layer).x(maxInd) *(3*selfOrgNet(num_layer).v_th)/(wink(1)+eps);
                else
                    selfOrgNet(num_layer).x(maxInd) = selfOrgNet(num_layer).x(maxInd).*(5*selfOrgNet(num_layer-1).v_th)./(wink(1)+eps);
                end

                % "Input" to LGN
                selfOrgNet(num_layer).x = sparse(selfOrgNet(num_layer).x); % Sparsity declaration for memory/speed

                % Solve Wave Dynamical System in LGN
                selfOrgNet(num_layer).fb = 1*selfOrgNet(num_layer).S*selfOrgNet(num_layer).H; 

                if num_layer ~= total_layers
                    selfOrgNet(num_layer).ff = 5*selfOrgNet(num_layer).S_d*selfOrgNet(num_layer).x; %  Change S_d *1/5 for V1;    
                else
                    selfOrgNet(num_layer).ff = selfOrgNet(num_layer).x; %  Change S_d *1/5 for V1;
                end

                selfOrgNet(num_layer).v = RK4(@(v)(-1./selfOrgNet(num_layer).tau_v .*v + selfOrgNet(num_layer).fb + selfOrgNet(num_layer).ff ),  dt,selfOrgNet(num_layer).v);
                selfOrgNet(num_layer).th = RK4(@(th)(1./selfOrgNet(num_layer).tau_th.*(selfOrgNet(num_layer).v_th-th).*(1-selfOrgNet(num_layer).H) + selfOrgNet(num_layer).th_plus.*selfOrgNet(num_layer).H),  dt,selfOrgNet(num_layer).th);
                
                if debugMode
                    if mod(tt,dt) == 0
                        figure(5),
                        cla()
                        plot(selfOrgNet(2).v)
                        hold on
                        plot(selfOrgNet(2).th)
                        plot(selfOrgNet(2).x)
                        hold off
                        sprintf('# of firing neurons:='), length(find(selfOrgNet(2).v >= selfOrgNet(2).th))
                        figure(6)
                        tmp = selfOrgNet(2).W'* selfOrgNet(1).Hint; 
                        hist(tmp)
                    end
                end
                
                % Discontinuous Update rule in LGN
                selfOrgNet(num_layer).fire = find(selfOrgNet(num_layer).v >= selfOrgNet(num_layer).th); 
                selfOrgNet(num_layer).v(selfOrgNet(num_layer).fire) = selfOrgNet(num_layer).v_reset(selfOrgNet(num_layer).fire);
                selfOrgNet(num_layer).H = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
                selfOrgNet(num_layer).H(selfOrgNet(num_layer).fire,1) = ones(length(selfOrgNet(num_layer).fire),1);
                
                if debugMode
                    if mod(tt,dt) == 0

                        %tt, lr
                        selfOrgNet(1).fire = find(selfOrgNet(1).H == 1); 
                        figure(2),
                        %title(lossVal(end))
                        for num_layer2 = 1:total_layers
                            num_layer = num_layer2;

                            subplot(1,total_layers,num_layer), 
                            title(sprintf('# firing neurons = %d',length(selfOrgNet(num_layer).fire)))
                            hold on,axis ij image, 
                            scatter(selfOrgNet(num_layer).nx(:,1),selfOrgNet(num_layer).nx(:,2),'k','filled'),...
                                                                    scatter(selfOrgNet(num_layer).nx(selfOrgNet(num_layer).fire,1),selfOrgNet(num_layer).nx(selfOrgNet(num_layer).fire,2),'r','filled')

                        end
                        pause(0.2)
                    end
                end
                
            end
            
%             error_net = (selfOrgNet(total_layers).H - selfOrgNet(total_layers).Hdes);
%             objSize = sum(selfOrgNet(total_layers).Hdes);
%             onTarget = length(find(error_net == -1))/objSize;
%             lossVal(end+1) = onTarget;
            
            % FEEDBACK (LIKE BACKPROP)
            if (STDP == true && mod(tt,STDP_dt) == 0)

%                 if debugMode2 
%                     figure(3)
%                     title('Before update')
%                     for num_layer = 2:total_layers
%                         subplot(1,total_layers-1,num_layer-1)
%                         imagesc(selfOrgNet(num_layer).W')
%                         colorbar
%                     end
%                 end
                
                % BACKPROP
                
                objSize = sum(selfOrgNet(total_layers).Hdes);
                %error_net = (selfOrgNet(total_layers).H - selfOrgNet(total_layers).Hdes)/objSize;
                error_net = (selfOrgNet(total_layers).H - selfOrgNet(total_layers).Hdes);
                selfOrgNet(total_layers).Herr= error_net;
                
                tmpError = error_net;
                objSize = sum(selfOrgNet(total_layers).Hdes);
                onTarget = length(find(error_net == -1))/objSize;
                offTarget = length(find(error_net == 1))/objSize; 
                
                lossVal(end+1) = (onTarget*10 + 5*offTarget)/15;
%                 lossVal(end+1) = onTarget;
%                 lossVal(end+1) = sum(abs(error_net));                
                lossVal2(end+1) = sum(abs(error_net));
                %length(lossVal2), tt, Tend
                for num_layer2 = total_layers:-1:2
                    num_layer = num_layer2;
                    update = error_net; 
                    
                    for num_layer3 = total_layers-1:-1:num_layer2
                        num_layer_inner = num_layer3;
                        update = (selfOrgNet(num_layer_inner+1).W)*update; 
                    end
                    tmp = (selfOrgNet(num_layer-1).Hint*(-1*update'));
                    %selfOrgNet(num_layer).W = selfOrgNet(num_layer).W + STDP_dt*(selfOrgNet(num_layer).eta_learn* (selfOrgNet(num_layer-1).Hint*(-1*update')));
                    if and(max(tmp(:))~=0, num_layer ==3)
                        selfOrgNet(num_layer).W = selfOrgNet(num_layer).W + STDP_dt*(selfOrgNet(num_layer).eta_learn* tmp)/max(tmp(:));
                    else
                        selfOrgNet(num_layer).W = selfOrgNet(num_layer).W + STDP_dt*(selfOrgNet(num_layer).eta_learn*(selfOrgNet(num_layer-1).H * selfOrgNet(num_layer).H'));
                    end
                    
                    %selfOrgNet(num_layer).W(:,selfOrgNet(num_layer).fire) = 10*(selfOrgNet(num_layer).W(:,selfOrgNet(num_layer).fire))./range(selfOrgNet(num_layer).W(:,selfOrgNet(num_layer).fire),1); 
                    selfOrgNet(num_layer).W = 5*selfOrgNet(num_layer).W./range(selfOrgNet(num_layer).W);
                    %selfOrgNet(num_layer).W(selfOrgNet(num_layer-1).fire,:) = 10*(selfOrgNet(num_layer).W(selfOrgNet(num_layer-1).fire,:))./range(selfOrgNet(num_layer).W(selfOrgNet(num_layer-1).fire,:),2); 
                end
                
                if debugMode2
                    figure(4)
                    title('after update')
                    for num_layer2 = 2:total_layers
			num_layer = num_layer2;
                        subplot(1,total_layers-1,num_layer-1)
                        imagesc(selfOrgNet(num_layer).W')
                        colorbar
                    end
                end
                
            end
            
            if debugMode2
                if mod(tt,STDP_dt) == 0   % Diagnostics
                    tt, lr
                    selfOrgNet(1,1).fire = find(selfOrgNet(1,1).H == 1); 
                    figure(2),
                    title(lossVal2(end))
                    for num_layer2 = 1:total_layers
			num_layer = num_layer2;

                        subplot(1,total_layers,num_layer), hold on,axis ij image, scatter(selfOrgNet(num_layer).nx(:,1),selfOrgNet(num_layer).nx(:,2),'k','filled'),...
                                                                scatter(selfOrgNet(num_layer).nx(selfOrgNet(num_layer).fire,1),selfOrgNet(num_layer).nx(selfOrgNet(num_layer).fire,2),'r','filled')

                    end
                    pause(0.2)
                end
            end
        
            if mod(tt,dt) == 0

                ts = ts+1;
                L2v(:,ts) = selfOrgNet(2).v;
                L2H(:,ts) = selfOrgNet(2).H;

                L3v(:,ts) = selfOrgNet(3).v;
                L3H(:,ts) = selfOrgNet(3).H;

    %             L4v(:,ts) = selfOrgNet(4).v;
    %             L4H(:,ts) = selfOrgNet(4).H;
            end        

        end
   
        toc
        disp('SNN run complete') 
        J = length(find(lossVal >0.8));
        
        J_vec = [J_vec J];
        lr_vec = [lr_vec lr]; 
        J_vec
        

        
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
%     
        sprintf('ss = %d',ss)
        netLossVal{1, ss} = lossVal;

        %%%%%%%%%%%%%%%% End of LR optimization loop
    %     fname_save = sprintf('autoenc2/autoenc_diffObjSamePath_column=%d_ss=%d',5,ss);
    % 	save(fname_save, 'selfOrgNet','netLossVal')
        
        if mod(ss, save_freq) == 0
            fname_save2 = strcat(fname_AEsave, sprintf('column=%d_ep=%d_test',colId,ss));
            %fname_save = sprintf('trainedNets/autoenc_column=%d_ep=%d',colId,ss);
            mysave(fname_save2, selfOrgNet, netLossVal, J_vec, lr_vec)
        end
        
%         if lr == 0
%             break
%         end
        
    end  
%     tot_epochs = [tot_epochs; ss];
    
%     fname_save2 = strcat(fname_AEsave, sprintf('column=%d',colId));    
%     %fname_save = sprintf('trainedNets/autoenc_column=%d',colId);
%     mysave(fname_save2, selfOrgNet, netLossVal, J_vec, lr_vec)

end
% opt.tot_epochs = tot_epochs;
% fname_optSave = strcat(folderSave, '/autoenc_opt.mat');
% save(fname_optSave,'opt')



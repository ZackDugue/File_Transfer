function [] = unsupervisedRun(X_input, label, total_columns, saveMovie, folderSave, fnameAE_save, fname_Actsave, save_freq)

% INPUTS


if saveMovie == true
    gifFile = input('Movie filename: ');
    debugMode2 = logical(1);
else
    gifFile = '';
    debugMode2 = logical(0);
end

% LOAD HYPERPARAMETERS OF NETWORK
optName = strcat(folderSave, '/autoenc_opt.mat');
opt = {};
load(optName)   


tot_epochs = opt.tot_epochs;
total_layers = opt.total_layers; %(Including input and output layer)

%for colId = 1:1%total_columns
parfor colId = 1:total_columns

    colId2 = colId; 
    t_ep = tot_epochs(colId2);
    %disp('printing in side parfor')
    %t_ep, colId2, opt.tot_epochs(colId2);
    
    for epoch2 = save_freq:save_freq:t_ep 
    epoch = epoch2;
    
    %%%% Time loop settings & Initialization of LR optimization
    % netName = sprintf('trainedNets/autoenc_column=%d.mat',colId);  
    netName = strcat(fnameAE_save, sprintf('column=%d_ep=%d.mat',colId, epoch));
    
    activity_Hint = cell(1,opt.total_layers-1);
    
    if ~exist(netName, 'file')
        continue;
    else
        selfOrgNet = myload(netName)
    end
%     if ~isfile(netName)
%         continue;
%     else
%         selfOrgNet = myload(netName)
%     end
    
    for ss = 1:1
        
        Tend = size(X_input,2); dt = 0.1; t_intrvls = 0:Tend/dt;  tx = 0; 
        
        %tmpY = [true, diff(label)~=0];
        %uni_tmpY = label(tmpY)'
        num_patterns = length(label);
        
        disp(sprintf("Number of patterns = %d",num_patterns));
        
        Trot = Tend/num_patterns; ts = 0;  
        
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
        STDP        = logical(0); STDP_dt = 5*dt; 
        opti_lr     = logical(0);
        state_reset = logical(1);
        debugMode   = logical(0);
        %debugMode2  = logical(1);

        L1v = zeros(selfOrgNet(1).numNeurons, Tend*10);
        L1H = zeros(selfOrgNet(1).numNeurons, Tend*10);
        L2v = zeros(selfOrgNet(2).numNeurons, Tend*10);
        L2H = zeros(selfOrgNet(2).numNeurons, Tend*10);
        L3v = zeros(selfOrgNet(3).numNeurons, Tend*10);
        L3H = zeros(selfOrgNet(3).numNeurons, Tend*10);
%         L4v = zeros(selfOrgNet(3).numNeurons, Tend*10);
%         L4H = zeros(selfOrgNet(3).numNeurons, Tend*10);
        
        lossVal = [];
        tic
        %Tend = 50;
        %t_st = 0;
        %tx = 0;
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
                        
                        
                        selfOrgNet(1).Hwin = sparse(zeros(selfOrgNet(num_layer).numNeurons,opt.nTwin(num_layer,colId)));
                        selfOrgNet(1).Hint = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
                        
                        for num_layer2 = 2:total_layers
                            num_layer = num_layer2;
                            
%                             disp(sprintf('Entering state-reset at num_layer=%d, tt=%d, tx=%d',num_layer, tt, tx));
                        
                            selfOrgNet(num_layer).Hwin = sparse(zeros(selfOrgNet(num_layer).numNeurons,opt.nTwin(num_layer,colId)));
                            selfOrgNet(num_layer).Hint = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
                            selfOrgNet(num_layer).Hdot = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); 
                            selfOrgNet(num_layer).Herr = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
                            selfOrgNet(num_layer).thresh = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); 
                            selfOrgNet(num_layer).v = 0*ones(selfOrgNet(num_layer).numNeurons,1);     
                            selfOrgNet(num_layer).H = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); 
                            selfOrgNet(num_layer).x = sparse(zeros(selfOrgNet(num_layer).numNeurons,1)); 
                            selfOrgNet(num_layer).th = ones(selfOrgNet(num_layer).numNeurons,1);                 
                            
%                             selfOrgNet(num_layer).v = zeros(selfOrgNet(num_layer).numNeurons,1);
%                             selfOrgNet(num_layer).H = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
%                             selfOrgNet(num_layer).x = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
%                             if num_layer ~= total_layers
%                                 selfOrgNet(num_layer).Hwin = sparse(zeros(selfOrgNet(num_layer).numNeurons,opt.nTwin(num_layer,colId))); 
%                             else
%                                 selfOrgNet(num_layer).Hwin = sparse(zeros(selfOrgNet(num_layer).numNeurons,opt.nTwin(num_layer,colId))); 
%                             end
%                             selfOrgNet(num_layer).Hint = sparse(zeros(selfOrgNet(num_layer).numNeurons,1));
%                             selfOrgNet(num_layer).th = 0*ones(selfOrgNet(num_layer).numNeurons,1);
                        end
                    end
                end
            end
               
            for num_layer2 = 2:total_layers
                num_layer = num_layer2;
                
%                 if mod(tt,Trot) == 0
%                     tx
%                     num_layer
%                     size(selfOrgNet(num_layer-1).Hwin)
%                     selfOrgNet(num_layer-1).Hint
% %                     if num_layer == 2
% %                         selfOrgNet(num_layer-1).Hint
% %                     end
%                         %selfOrgNet(num_layer-1).Hint
%                 end
                
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
                        figure(4),
                        cla()
                        plot(selfOrgNet(2).v)
                        hold on
                        plot(selfOrgNet(2).th)
                        plot(selfOrgNet(2).x)
                        hold off
                        sprintf('# of firing neurons:='), length(find(selfOrgNet(2).v >= selfOrgNet(2).th))

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
                        cla()
                        
                        plot(selfOrgNet(2).v)
                        
%                         for num_layer2 = 1:total_layers
% 
%                             num_layer = num_layer2;
%                             subplot(1,total_layers,num_layer), 
%                             title(sprintf('# firing neurons = %d',length(selfOrgNet(num_layer).fire)))
%                             hold on,axis ij image, 
%                             scatter(selfOrgNet(num_layer).nx(:,1),selfOrgNet(num_layer).nx(:,2),'k','filled'),...
%                                                                     scatter(selfOrgNet(num_layer).nx(selfOrgNet(num_layer).fire,1),selfOrgNet(num_layer).nx(selfOrgNet(num_layer).fire,2),'r','filled')
% 
%                         end
                        pause(0.2)
                    end
                end
                
            end
            

            if debugMode2
                filename = strcat(gifFile,sprintf('_column=%d_ep=%d.gif',colId, epoch));
    
                if mod(tt,2) == 0   % Diagnostics
                    selfOrgNet(1,1).fire = find(selfOrgNet(1,1).H == 1); 
                    h=figure('visible','off');
    %                 title(lossVal(end))
                    for num_layer2 = 1:total_layers
                        num_layer = num_layer2;

                        subplot(1,total_layers,num_layer), hold on,axis ij image, scatter(selfOrgNet(num_layer).nx(:,1),selfOrgNet(num_layer).nx(:,2),'k','filled'),...
                                                                scatter(selfOrgNet(num_layer).nx(selfOrgNet(num_layer).fire,1),selfOrgNet(num_layer).nx(selfOrgNet(num_layer).fire,2),'r','filled')

                    end
                    pause(0.2)

                    frame = getframe(h); 
                    im = frame2im(frame); 
                    [imind,cm] = rgb2ind(im,256); 
                    % Write to the GIF File 
                    if tt == 0 
                        imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
                    else 
                        tt
                        imwrite(imind,cm,filename,'gif','WriteMode','append'); 
                    end 
                end
            end
        
            if mod(tt,dt) == 0

                ts = ts+1;
                
                L1v(:,ts) = selfOrgNet(1).v;
                L1H(:,ts) = selfOrgNet(1).H;

                
                L2v(:,ts) = selfOrgNet(2).v;
                L2H(:,ts) = selfOrgNet(2).H;

                L3v(:,ts) = selfOrgNet(3).v;
                L3H(:,ts) = selfOrgNet(3).H;
                
                activity_Hint{1} = [activity_Hint{1}, selfOrgNet(1).Hint];
                activity_Hint{2} = [activity_Hint{2}, selfOrgNet(2).Hint];
                

    %             L4v(:,ts) = selfOrgNet(4).v;
    %             L4H(:,ts) = selfOrgNet(4).H;

            end        

        end
   
        toc
        disp('SNN run complete') 
        
    end  
    
    % fname_save = sprintf('trainedNets/Act_column=%d',colId);
    fname_act_save2 = strcat(fname_Actsave, sprintf('column=%d_ep=%d', colId,epoch));
    mysave2(fname_act_save2, selfOrgNet, L1v, L2v, L3v, L1H, L2H, L3H, activity_Hint)
    
    end

end

function [] = plotConnectionMatrix(unsupervised, supervised, folderSave, fname_AEsave, total_columns, save_freq)


% LOAD HYPERPARAMETERS OF NETWORK
optName = strcat(folderSave, '/autoenc_opt.mat');
opt = {};
load(optName)   

tot_epochs = opt.tot_epochs;
total_layers = opt.total_layers; %(Including input and output layer)

if unsupervised == true
   
    for colId = 1:total_columns
        
        %for ep = save_freq:save_freq:tot_epochs(colId)
        ep = tot_epochs(colId);
        
        %column = 1; ep = 20;
        AE_name = strcat(fname_AEsave, sprintf('column=%d_ep=%d.mat',colId, ep));
        
        if ~isfile(AE_name)
            while(1)
                ep = ep - 1;
                AE_name = strcat(fname_AEsave, sprintf('column=%d_ep=%d.mat',colId, ep));
                if isfile(AE_name) > 0
                    load(AE_name)
                    break
                end
            end
            %continue;
        else
            load(AE_name)
        end
        
        W1 = selfOrgNet(2).W;
        rndNeurons = datasample(1:size(W1,2),9,'Replace',false);
        L1pos = selfOrgNet(1).nx;
        h = figure; 
        title(sprintf('Column-%d',colId))
        for ii = 1:length(rndNeurons)
            subplot(3,3,ii)
            title(sprintf('Neuron - %d',rndNeurons(ii)))
            scatter(L1pos(:,1), L1pos(:,2),[],W1(:,rndNeurons(ii)),'filled')
            set(gca,'visible','off')
        end
        
        figFolder = strcat(folderSave, '/PerformanceFig/');

        if ~exist(figFolder, 'dir')
           mkdir(figFolder)
        end

        figName = strcat(figFolder, sprintf('connectMatrix_col=%d_ep=%d.fig',colId, ep));
        savefig(h, figName)
        
        %end
        
    end   
    
end
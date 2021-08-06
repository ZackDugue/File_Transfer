function [] = plotTrajectory(unsupervised, supervised, folderSave, X_input, labels_all, total_columns, total_epochs, save_freq)

    if unsupervised == true
        
        num_classes = length(unique(labels_all));
        num_ex_eachClass = length(find(labels_all==1));
        
        for colId = 1:total_columns
            
            for ss = 1:2%total_epochs

                if mod(ss, save_freq) == 0
                    
                    L2v_all =[];

                    fname_act = strcat(folderSave, '/Act/Act_');
                    fname_act = strcat(fname_act, sprintf('column=%d_ep=%d.mat',colId, ss));
                    load(fname_act, 'L2v', 'L2H')
                    L2v_all = [L2v_all, L2v];
                    
                    fname_test_act = strcat(folderSave, '/Act_test/Act_');
                    fname_test_act = strcat(fname_act, sprintf('column=%d_ep=%d.mat',colId, ss));
                    load(fname_act, 'L2v', 'L2H')
                    L2v_all = [L2v_all, L2v];
                    
                    size(L2v_all)
                    [coeff,score,latent,tsquared,explained,mu] = pca(L2v_all');
                    h = figure(1); 
                    
                    disp('score size')
                    size(score)
                    
                    for start = 1:num_ex_eachClass*10:size(X_input,2)*10
                        stop = start + num_ex_eachClass*10 - 1; mid = 1;
                        start, stop
                        plot3(score(start:mid:stop,1),score(start:mid:stop,2),score(start:mid:stop,3))
                        hold on
                    end
                    
%                     for start = size(X_input,2)*10+1: num_ex_eachClass*10: 2*size(X_input,2)*10
%                         stop = start + num_ex_eachClass*10 - 1; mid = 1;
%                         start, stop
%                         plot3(score(start:mid:stop,1),score(start:mid:stop,2),score(start:mid:stop,3))
%                         hold on
%                     end
%                     legend('circle-train', 'square-train','circle-test','square-test')
                    legend('circle', 'square')
                    set(gca, 'XTick', [])
                    set(gca, 'YTick', [])
                    set(gca, 'ZTick', [])
                    
                    xlabel('PC-1')
                    ylabel('PC-2')
                    zlabel('PC-3')
                    
                    figName = strcat(folderSave, '/TrajectoryFig/');
                    
                    if ~exist(figName, 'dir')
                       mkdir(figName)
                    end
                    
                    
                    figName = strcat(figName, sprintf('L2v_column=%d_ep=%d.fig',colId, ss));
                    savefig(h, figName)
                    close all
                end
            end
        end            
    end
    
    
    if supervised == true
        
        num_classes = length(unique(labels_all));
        num_ex_eachClass = length(find(labels_all==1));
        
        for colId = 1:total_columns
            
            for ss = 1:total_epochs

                if mod(ss, save_freq) == 0

                    fname_act = strcat(folderSave, '/S_train_');
                    fname_act = strcat(fname_act, sprintf('column=%d_ep=%d.mat',colId, ss));
                    load(fname_act, 'L2v', 'L2H')

                    [coeff,score,latent,tsquared,explained,mu] = pca(L2v');
                    h = figure(1); 
                    

                    for start = 1:num_ex_eachClass*10:size(X_input,2)*10
                        stop = start + num_ex_eachClass*10 - 1; mid = 1;
                        plot3(score(start:mid:stop,1),score(start:mid:stop,2),score(start:mid:stop,3))
                        hold on
                    end

                    figName = strcat(folderSave, '/TrajectoryFig/');
                    
                    if ~exist(figName, 'dir')
                       mkdir(figName)
                    end                    
                    
                    figName = strcat(figName, sprintf('L2v_column=%d_ep=%d.fig',colId, ss));
                    savefig(h, figName)
                    close all
                end
            end
        end            
    end    
    
    
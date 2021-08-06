function [] = plotPerformance(unsupervised, supervised, folderSave, total_columns, total_epochs, num_classes, acc_train, acc_test)

if unsupervised == true

    %tot_ep = size(acc_train{1},2);
    
    tot_ep = 0;
    for ii = 1:total_columns
        disp('in loop')
        size(acc_train{ii},2)
        if tot_ep < size(acc_train{ii},2)
            tot_ep = size(acc_train{ii},2);
        end
    end
    tot_ep
    
    accVal_trainRT = zeros(total_columns, tot_ep)*NaN;
    
    for colId = 1:total_columns
        colId
        tmp = [acc_train{colId}{:}];
        accVal_trainRT(colId, 1:length(tmp(1:2:end))) = tmp(1:2:end);
    end
    
    %tot_ep = size(acc_test{1},2);
    tot_ep = 0;
    for ii = 1:total_columns
        if tot_ep < size(acc_test{ii},2)
            tot_ep = size(acc_test{ii},2);
        end
    end
    
    
    accVal_testRT = zeros(total_columns, tot_ep)*NaN;
    
    for colId = 1:total_columns
        tmp = [acc_test{colId}{:}];
        accVal_testRT(colId, 1:length(tmp(1:2:end))) = tmp(1:2:end);
    end
    
    
    h = figure(1);
    
    if total_columns == 1
        errorbar(1:tot_ep, mean(accVal_trainRT, 'omitnan'), std(accVal_trainRT,'omitnan'), 'b-')
        hold on
        errorbar(1:tot_ep, mean(accVal_testRT, 'omitnan'), std(accVal_testRT, 'omitnan'), 'r-')
    else
        h1 = figure(1);
        errorbar(1:tot_ep, mean(accVal_trainRT, 'omitnan'), std(accVal_trainRT, 'omitnan'), 'b-')
        hold on
        tmp = [acc_train{end}{:}];
        plot(1:length(tmp(1:2:end)), tmp(1:2:end), 'r-')
        
        h2 = figure(2);
        errorbar(1:tot_ep, mean(accVal_testRT,'omitnan'), std(accVal_testRT, 'omitnan'), 'b')
        hold on
        tmp = [acc_test{end}{:}];
        plot(1:length(tmp(1:2:end)), tmp(1:2:end), 'r-')        
    end
    
    figName = strcat(folderSave, '/PerformanceFig/');

    if ~exist(figName, 'dir')
       mkdir(figName)
    end

    figName = strcat(figName, 'US_columnPerf1_train.fig');
    savefig(h1, figName)
    
    figName = strcat(figName, 'US_columnPerf1_test.fig');
    savefig(h2, figName)
    %close all
    
    %%%%%%%%% ----- acc_columns2 ---- %%%%%%%%%%%
    
%     tot_ep = size(acc_test{1},2);
%     accVal_testRT = zeros(total_columns, tot_ep);
%     
%     for colId = 1:total_columns
%         tmp = [acc_test{colId}{:}];
%         accVal_testRT(colId, :) = tmp(1:2:end);
%     end
%     
%     h = figure(1);
%     
%     if total_columns == 1
%         errorbar(1:tot_ep, mean(accVal), std(accVal), 'b')
%     else
%         errorbar(1:tot_ep, mean(accVal), std(accVal), 'b')
%         hold on
%         plot(1:tot_ep, acc_test{end}, 'r')
%     end
%     
%     figName = strcat(folderSave, '/PerformanceFig/');
% 
%     if ~exist(figName, 'dir')
%        mkdir(figName)
%     end
% 
%     figName = strcat(figName, 'US_columnPerf2.fig');
%     savefig(h, figName)

    
end

if supervised == true

    acc1_ctr = zeros(total_columns, total_epochs);
    acc1_train = zeros(total_columns, total_epochs);
    
    acc2_ctr = zeros(total_columns, total_epochs);
    acc2_train = zeros(total_columns, total_epochs);
    
    for colId = 1:total_columns

        fname_train = strcat(folderSave, '/S_train_');
        fname_train = strcat(fname_train, sprintf('column=%d.mat',colId));
        load(fname_train, 'acc_ctrl', 'acc_train')
        tot_save = length(acc_ctrl);
        
        for ss = 1:length(acc_ctrl)
            acc1_ctr(colId, ss) = acc_ctrl{ss}(1)/num_classes*100;
            acc2_ctr(colId, ss) = acc_ctrl{ss}(2)*100;
        end
        
        for ss = 1:length(acc_train)
            acc1_train(colId, ss) = acc_train{ss}(1)/num_classes*100;            
            acc2_train(colId, ss) = acc_train{ss}(2)*100;
        end
        
    end
    
    figure; 
    
    h1 = figure(1);    
    if total_columns == 1
        plot(1:length(acc_train), mean(acc1_train(1:length(acc_train)),1))
    else
        errorbar(1:length(acc_train), mean(acc1_train(1:length(acc_train)),1), std(acc1_train(1:length(acc_train)),[],1))    
    end
    
    h2 = figure(2);
    
    if total_columns == 1
        x = 1:length(acc_train);
        y = mean(acc2_train(1:length(acc_train)),1);
        plot(x,y)
        %plot(1:length(acc_train), mean(acc2_train(1:length(acc_train)),1))
    else        
        errorbar(1:length(acc_train), mean(acc2_train(1:length(acc_train)),1), std(acc2_train(1:length(acc_train)),[],1))
    end

    figName = strcat(folderSave, '/PerformanceFig/');

    if ~exist(figName, 'dir')
       mkdir(figName)
    end

    figName1 = strcat(figName, 'cumulative_gesturePerf.fig');
    figName2 = strcat(figName, 'RT_gesturePerf.fig');
    
    savefig(h1, figName1)
    savefig(h2, figName2)
    
end



    


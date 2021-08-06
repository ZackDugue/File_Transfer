
function [] = multiColumn_SNN(dataset, total_columns, unsupervised, supervised, taskSwitch, motionPredict, folderSave, trajectoryPlot, performancePlot, to_test,num_layers)

%% INPUT ARGUMENTS

% dataset: Name of dataset of dynamic scene
% total_columns: Number of columns in SNN architecture
% unsupervised: Unsupervised training followed by linear classifier 
% supervised: Direct supervised training
% trajectoryPlot: Plot the dynamic trajectory (after unsupervised training)
% performancePlot: Plot the performance of the network (on the training sample

%% Load dataset with dynamic scene
% load(dataset)
% X_input = data_all; 
% 
% X_train = X_input;
% labels_all_train = labels_all;
% 
% if exist('label','var') ~= 1
%     label = unique(labels_all);
% end
% label_train = label;
% 
% acc_columns1 = cell(1,total_columns+1);
% acc_columns2 = cell(1,total_columns+1);

%% UNSUPERVISED LEARNING COUPLED WITH DISCRIMINATOR TO LEARN LABELS
if unsupervised == true

    num_datasets = length(dataset);
    %num_datasets = 1;
    acc_train = cell(1,total_columns+1);
    acc_test = cell(1,total_columns+1);
    p_kSave_train = cell(1,total_columns+1);
    p_kSave_test = cell(1,total_columns+1);
    RT_trainMat = cell(1, total_columns+1);
    RT_testMat = cell(1, total_columns+1);
    
    acc_eachClass_train = cell(1,total_columns+1);
    acc_eachClass_test = cell(1,total_columns+1);
    
    % Unsupervised training
    
    total_epochs = 20;
    save_freq = 2;
    
    if ~exist(folderSave, 'dir')
       mkdir(folderSave)
    end
    
    folder_AEsave = strcat(folderSave, '/AE');
    if ~exist(folder_AEsave, 'dir')
       mkdir(folder_AEsave)
    end
    fname_AEsave = strcat(folder_AEsave, '/AE_');
    
    
    % LOAD ALL DATASETS
    
    X_train = {};
    labels_all_train = {};
    label_train = {};

    for ii = 1:num_datasets
        load(dataset{ii})
        X_train{ii} = data_all; 
        labels_all_train{ii} = labels_all;
        if exist('label','var') ~= 1
            tmpY = [true, diff(labels_all_train{ii})~=0];
            uni_tmpY = labels_all_train{ii}(tmpY);
            num_patterns = length(uni_tmpY);
            %label_train{ii} = unique(labels_all, 'stable');
            label_train{ii} = uni_tmpY;
            %num_patterns
        else
            label_train{ii} = label;
        end
    end   
    

    % CONCATENATE ALL DATASETS
    X_input = [];
    for ii = 1:1%num_datasets
        X_input = [X_input, X_train{ii}];
    end
    
    %unsupervisedTrain2_test(X_input, total_columns, total_epochs, save_freq, folderSave, fname_AEsave);
    %unsupervisedTrain_test(X_input, total_columns, total_epochs, save_freq, folderSave, fname_AEsave);

    unsupervisedTrain2(X_input, total_columns, total_epochs, save_freq, folderSave, fname_AEsave,num_layers);
    %unsupervisedTrain(X_input, total_columns, total_epochs, save_freq, folderSave, fname_AEsave);

    % Run unsupervised network on same dataset
    %saveMovie = input('Save movie? (true/false): ');
    saveMovie = false;
    
    folder_Actsave = strcat(folderSave, '/Act');
    if ~exist(folder_Actsave, 'dir')
       mkdir(folder_Actsave)
    end
    fname_Actsave = strcat(folder_Actsave, '/Act_');
    
    % TRAIN DATASET
    X_input = X_train{1};

    disp('begin unsupervisedRun')
    unsupervisedRun(X_input, label_train{1}, total_columns, saveMovie, folderSave, fname_AEsave, fname_Actsave, save_freq)

    if to_test == true
        
        X_test = X_train{2};
        labels_all_test = labels_all_train{2};
        label_test = label_train{2};
        
        %saveMovie_test = input('Save test movie? (true/false): ');
        saveMovie_test = false;
        save_freq_test = save_freq; 
        
        folder_ActTest = strcat(folderSave, '/Act_test');
        if ~exist(folder_ActTest, 'dir')
           mkdir(folder_ActTest)
        end
        
        fname_ActTest_save = strcat(folder_ActTest, '/Act_');
        % unsupervised run over test dataset
        unsupervisedRun(X_test, label_train{2}, total_columns, saveMovie_test, folderSave, fname_AEsave, fname_ActTest_save, save_freq_test)
        
    else
        fname_ActTest_save = '';
    end
    
    for ii = 1:num_datasets
        labels_RT = labels_all_train{ii};
        labels_RT = repelem(labels_RT, 10);
        %labels_RT2 = categorical(labels_RT');
        %labels_RT2 = onehotencode(labels_RT2, 2);
        %labels_RT = labels_RT2';
        save(strcat(folderSave, sprintf('/RT_labels-%d.mat',ii)), 'labels_RT');
    end
    
    
    
%     Append discriminator and classify
    parfor jj2 = 1:total_columns+1
        jj = jj2; 
        
        if jj>total_columns
            columns_interest = 1:total_columns;
        else
            columns_interest = jj;
        end
        
%         labels_all = labels_all_train{1};
%         X_input = X_train{1};
%         label = label_train{1};
        
        [accVal1, accVal2, p_k_train, p_k_test, RT_train, RT_test] = discriminator(X_train, labels_all_train, label_train, columns_interest, folderSave, fname_Actsave, fname_ActTest_save, save_freq, to_test);
        
        acc_train{jj2} = accVal1;
        acc_test{jj2} = accVal2;
        p_kSave_train{jj2} = p_k_train;
        p_kSave_test{jj2} = p_k_test;
        RT_trainMat{jj2} = RT_train;
        RT_testMat{jj2} = RT_test;
        
        
    end
    
    fname_dsrm = strcat(folderSave, '/DiscrimPerf');
    save(fname_dsrm, 'acc_train', 'acc_test', 'p_kSave_train', 'p_kSave_test', 'RT_trainMat', 'RT_testMat')
   
    
%     % Evaluate specialization of the columns
%     spec_score_train = specialization_scorer(p_kSave_train);
%     spec_score_test = specialization_scorer(p_kSave_test);
%     
%     fname_spec = strcat(folderSave, '/spec_score');
%     save(fname_spec, 'spec_score_train', 'spec_score_test')
end

%% NEW PARADIGM FOR TASK SWITCHING
if taskSwitch == true 
    
    num_datasets = length(dataset);

%     acc_columns1 = cell(num_datasets, total_columns+1);
%     acc_columns2 = cell(num_datasets, total_columns+1);

    total_epochs = input('Number of unsupervised training epochs (~1-100): ');
    save_freq = input('Frequency of saving network checkpoints (1-5): ');
    
    if ~exist(folderSave, 'dir')
       mkdir(folderSave)
    end
    
    folder_AEsave = strcat(folderSave, '/AE');
    if ~exist(folder_AEsave, 'dir')
       mkdir(folder_AEsave)
    end
    fname_AEsave = strcat(folder_AEsave, '/AE_');

    
    % LOAD ALL DATASETS
    
    X_train = {};
    labels_all_train = {};
    label_train = {};

    for ii = 1:num_datasets
        load(dataset{ii})
        X_train{ii} = data_all; 
        labels_all_train{ii} = labels_all;
        if exist('label','var') ~= 1
            label_train{ii} = unique(labels_all);
        else
            label_train{ii} = label;
        end
    end   
    
    % CONCATENATE ALL DATASETS
    X_input = [];
    for ii = 1:num_datasets
        X_input = [X_input, X_train{ii}];
    end
    
    
    % Unsupervised training
    %unsupervisedTrain(X_input, total_columns, total_epochs, save_freq, folderSave, fname_AEsave);
    
    % UNSUPERVISED RUN FOR ALL ENCODER TRAINED EPOCHS FOR ALL DATASETS
    for rr = 1:num_datasets

        X_input = X_train{rr};
        labels_all = labels_all_train{rr};
        label = label_train{rr};
        saveMovie = false;
        fname_act_save = strcat(folderSave, sprintf('/Act_dset=%d',rr));
        if ~exist(fname_act_save, 'dir')
           mkdir(fname_act_save)
        end
        fname_act_save = strcat(fname_act_save, '/Act_');

        % UNSUPERVISED RUN THRU TRAINED AUTOENC 
        unsupervisedRun(X_input, total_columns, saveMovie, folderSave, fname_AEsave, fname_act_save, save_freq)
        %unsupervisedRun(X_input, total_columns, saveMovie, fname_AE_save, fname_act_save, save_freq)

    end
    
    [X, Y] = meshgrid(1:num_datasets, 1:total_columns+1);
    group_var = [X(:), Y(:)];
    
    acc_columns1 = cell(1, length(group_var));
    acc_columns2 = cell(1, length(group_var));
    
    parfor gVar = 1:size(group_var,1)
                
        dset = group_var(gVar,1);
        colId = group_var(gVar,2);
        
        X_input = X_train{dset};
        labels_all = labels_all_train{dset};
        label = label_train{dset};
        
        fname_act_save = strcat(folderSave, sprintf('/Act_dset=%d/Act_',dset));
        
        if colId>total_columns
            columns_interest = 1:total_columns;
        else
            columns_interest = colId;
        end
        
        [accVal1, accVal2] = discriminator_taskSwitch(X_input, labels_all, label, columns_interest, folderSave, fname_act_save, save_freq);
        
        acc_columns1{gVar} = accVal1;
        acc_columns2{gVar} = accVal2;
        
        %acc_columns1{dset, colId} = accVal1;
        %acc_columns2{dset, colId} = accVal2;
        
    end
    
    acc1 = cell(num_datasets, total_columns+1);
    acc2 = cell(num_datasets, total_columns+1);
    
    for gVar = 1:size(group_var,1)
                
        dset = group_var(gVar,1);
        colId = group_var(gVar,2);
     
        acc1{dset, colId} = acc_columns1{gVar};
        acc2{dset, colId} = acc_columns2{gVar};
        
    end
        
    fname_dsrm = strcat(folderSave, '/DiscrimPerf');
    save(fname_dsrm, 'acc_columns1', 'acc_columns2', 'acc1','acc2')

end

%% UNSUPERVISED LEARNING COUPLED WITH DISCRIMINATOR TO LEARN LABELS
if motionPredict == true

    num_datasets = length(dataset);
    acc_train = cell(1,total_columns+1);
    acc_test = cell(1,total_columns+1);
    
    % Unsupervised training
    
    total_epochs = input('Number of unsupervised training epochs (~1-100): ');
    save_freq = input('Frequency of saving network checkpoints (1-5): ');
    
    if ~exist(folderSave, 'dir')
       mkdir(folderSave)
    end
    
    folder_AEsave = strcat(folderSave, '/AE');
    if ~exist(folder_AEsave, 'dir')
       mkdir(folder_AEsave)
    end
    fname_AEsave = strcat(folder_AEsave, '/AE_');
    
    
    % LOAD ALL DATASETS
    
    X_train = {};
    labels_all_train = {};
    label_train = {};

    for ii = 1:num_datasets
        load(dataset{ii})
        X_train{ii} = data_all; 
        labels_all_train{ii} = labels_all;
        if exist('label','var') ~= 1
            label_train{ii} = unique(labels_all);
        else
            label_train{ii} = label;
        end
    end   
    
    % CONCATENATE ALL DATASETS
    X_input = [];
    for ii = 1:num_datasets
        X_input = [X_input, X_train{ii}];
    end    
    
    unsupervisedTrain_motionPredict(X_input, total_columns, total_epochs, save_freq, folderSave, fname_AEsave);
    
    % Run unsupervised network on same dataset
    saveMovie = input('Save movie? (true/false): ');
    %saveMovie = false;
    
    folder_Actsave = strcat(folderSave, '/Act');
    if ~exist(folder_Actsave, 'dir')
       mkdir(folder_Actsave)
    end
    fname_Actsave = strcat(folder_Actsave, '/Act_');
    
    % TRAIN DATASET
    X_input = X_train{1};

    disp('begin unsupervisedRun')
    unsupervisedRun(X_input, total_columns, saveMovie, folderSave, fname_AEsave, fname_Actsave, save_freq)


end

%% SUPERVISED LEARNING FROM SCRATCH (over all layers)

% if supervised == true
% 
%     num_datasets = length(dataset);
%     
%     total_epochs = input('Number of supervised training epochs (~1-100): ');
%     save_freq = input('Frequency of saving network checkpoints (1-5): ');
%     
%     if ~exist(folderSave, 'dir')
%        mkdir(folderSave)
%     end
%     
%     % LOAD ALL DATASETS
%     
%     X_train = {};
%     labels_all_train = {};
%     label_train = {};
% 
%     for ii = 1:num_datasets
%         load(dataset{ii})
%         X_train{ii} = data_all; 
%         labels_all_train{ii} = labels_all;
%         if exist('label','var') ~= 1
%             label_train{ii} = unique(labels_all);
%         else
%             label_train{ii} = label;
%         end
%     end   
%     
%     fname_save = strcat(folderSave, '/S_train_');
%     supervisedTrain(X_train, labels_all_train, label_train, total_columns, total_epochs, save_freq, folderSave, fname_save, to_test);
%     
% end
% 
% fname_acc = strcat(folderSave, '/DiscrimPerf.mat');
% load(fname_acc);
% 
% if trajectoryPlot == true
%     plotTrajectory(unsupervised, supervised, folderSave, X_input, labels_all, total_columns, total_epochs, save_freq)
% end
% 
% 
% if performancePlot == true    
%     num_classes = length(unique(labels_all));
%     plotPerformance(unsupervised, supervised, folderSave, total_columns, total_epochs, num_classes, acc_train, acc_test)
% end
% 
% connectionMatrixPlot = false;
% if connectionMatrixPlot == true    
%     plotConnectionMatrix(unsupervised, supervised, folderSave, fname_AEsave, total_columns, save_freq)
% end
% %     fname_AEsave = '2MovingObj_test_hebb/AE/AE_';
% %     column = 1; ep = 20;
% %     AE_name = strcat(fname_AEsave, sprintf('column=%d_ep=%d.mat',column, ep));
% %     load(AE_name)
% %     W1 = selfOrgNet(2).W;
% %     rndNeurons = datasample(1:size(W1,2),9,'Replace',false);
% %     L1pos = selfOrgNet(1).nx;
% %     figure; 
% %     for ii = 1:length(rndNeurons)
% %         subplot(3,3,ii)
% %         title(sprintf('Neuron - %d',rndNeurons(ii)))
% %         scatter(L1pos(:,1), L1pos(:,2),[],W1(:,rndNeurons(ii)),'filled')
% %         set(gca,'visible','off')
% %     end
%     
%     

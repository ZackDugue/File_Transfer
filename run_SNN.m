
% PARAMETERS

dataset = {'inputDatasets/snapShot_2patternsN_train.mat', 'inputDatasets/snapShot_2patternsN_test-1.mat'};
total_columns = 2;
unsupervised = true;
supervised = false; 
folderSave='2patternsN_Hebb';
trajectoryPlot=true;
performancePlot=true;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 3 moving objects (2 diff traj, 1 same traj - diff shapes)
dataset = {'inputDatasets/snapShot_9patternsN_train.mat', 'inputDatasets/snapShot_9patternsN_test-1.mat'};
total_columns = 50;
unsupervised = true;
supervised = false; 
folderSave='9patternsN_nonhebb';
trajectoryPlot=true;
performancePlot=false;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 2 patterns -- 2 objects along 1 trajectory
dataset = {'inputDatasets/snapShot_4patternsN_train.mat', 'inputDatasets/snapShot_4patternsN_test-1.mat'};
total_columns = 2;
unsupervised = true;
supervised = false; 
folderSave='4patternsN_Hebb';
trajectoryPlot=true;
performancePlot=false;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 8 patterns -- 4 objects along 4 trajectory
dataset = {'inputDatasets/snapShot_4patternsN2_train.mat', 'inputDatasets/snapShot_4patternsN2_test-1.mat'};
total_columns = 2;
unsupervised = true;
supervised = false; 
folderSave='4+4patternsN2_Hebb';
trajectoryPlot=true;
performancePlot=false;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 16 different patterns
dataset = {'inputDatasets/snapShot_16patterns_train.mat'};
total_columns = 10;
unsupervised = true;
supervised = false; 
folderSave='16patterns';
trajectoryPlot=true;
performancePlot=false;
to_test = false;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 4 different patterns
dataset = {'inputDatasets/snapShot_4patterns_train.mat', 'inputDatasets/snapShot_4patterns_test_vel-2.mat' };
total_columns = 10;
unsupervised = true;
supervised = false; 
folderSave='4patterns';
trajectoryPlot=true;
performancePlot=false;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 25 different patterns
dataset = {'inputDatasets/snapShot_25patternsN_train.mat', 'inputDatasets/snapShot_25patternsN_test-1.mat' };
total_columns = 25;
unsupervised = true;
supervised = false; 
folderSave='25patternsN_hebb';
trajectoryPlot=true;
performancePlot=false;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 36 different patterns
dataset = {'inputDatasets/snapShot_36patternsN_train.mat', 'inputDatasets/snapShot_36patternsN_test-1.mat' };
total_columns = 25;
unsupervised = true;
supervised = false; 
folderSave='36patternsN_hebb';
trajectoryPlot=true;
performancePlot=false;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 16 different patterns
dataset = {'inputDatasets/snapShot_16patterns_train.mat', 'inputDatasets/snapShot_16patterns_test_vel-2.mat' };
total_columns = 10;
unsupervised = true;
supervised = false; 
folderSave='16patterns';
trajectoryPlot=true;

%% 16 patterns -- 4 objects along 4 trajectory
dataset = {'inputDatasets/snapShot_9patternsN_train.mat', 'inputDatasets/snapShot_9patternsN_test-4.mat'};
total_columns = 2;
unsupervised = true;
supervised = false; 
folderSave='9patternsN_Hebbperm_2col';
trajectoryPlot=false;
performancePlot=false;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);


%% 16 patterns -- 4 objects along 4 trajectory
dataset = {'inputDatasets/snapShot_8patternsN_train.mat', 'inputDatasets/snapShot_8patternsN_testNoise-1.mat'};
total_columns = 6;
unsupervised = true;
supervised = false; 
folderSave='8patternsN_NonHebb_6col';
trajectoryPlot=false;
performancePlot=false;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 15 different patterns
dataset = {'inputDatasets/snapShot_15patterns_train.mat', 'inputDatasets/snapShot_15patterns_test_vel-1.mat' };
total_columns = 50;
unsupervised = true;
supervised = false; 
folderSave='15patterns_hebb';
trajectoryPlot=true;
performancePlot=false;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 2moving obj small

dataset = {'inputDatasets/2movingObj_small.mat'};
total_columns = 1;
unsupervised = true;
supervised = false; 
folderSave='2movingObj_sm';
trajectoryPlot=false;
performancePlot=false;
to_test = false;
task_switch = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, folderSave, trajectoryPlot, performancePlot, to_test);

%% Unsupervise train - gesture
dataset = {'inputDatasets/gesture_train','inputDatasets/gesture_test'};
total_columns = 6;
unsupervised = true;
supervised = false; 
folderSave='gesture_US_Hebb';
trajectoryPlot=true;
performancePlot=true;
to_test = true;
task_switch = false;
motion_predict= false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% Unsupervise train - gesture
parpool(20)
dataset = {'inputDatasets/gesture_train','inputDatasets/gesture_test'};
total_columns = 20;
unsupervised = true;
supervised = false; 
folderSave='gesture_US_nonHebb';
trajectoryPlot=true;
performancePlot=true;
to_test = true;
task_switch = false;
motion_predict= false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% Unsupervise train - gesture
%parpool(50)
dataset = {'inputDatasets/gesture_train','inputDatasets/gesture_test'};
total_columns = 20;
unsupervised = true;
supervised = false; 
folderSave='gesture_US_Hebb_noAdj';
trajectoryPlot=true;
performancePlot=true;
to_test = true;
task_switch = false;
motion_predict= false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);


%% supervise train - gesture

dataset = 'inputDatasets/gesture_train';
total_columns = 1;
unsupervised = false;
supervised = true; 
folderSave='gesture';
trajectoryPlot=true;
performancePlot=false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, folderSave, trajectoryPlot, performancePlot);

%% supervise train - moving dots

dataset = 'inputDatasets/varyCo_data_test/varyCo8_70.mat';
total_columns = 1;
unsupervised = false;
supervised = true; 
folderSave='movingDots1';
trajectoryPlot=false;
performancePlot=true;
to_test = true;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, folderSave, trajectoryPlot, performancePlot, to_test);

test = 'inputDatasets/varyCo_data_train/varyCo_70.mat';

%% unsupervise train - movingDots

dataset = 'inputDatasets/varyCo_data_train/varyCo_70.mat';
total_columns = 1;
unsupervised = true;
supervised = false; 
folderSave='movingDots2';
trajectoryPlot=false;
performancePlot=true;
to_test = true;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, folderSave, trajectoryPlot, performancePlot, to_test);

test = 'inputDatasets/varyCo_data_train/varyCo_70.mat';

%% unsupervise train - moving dots

dataset = 'inputDatasets/varyCo_data_train/varyCo_70.mat';
total_columns = 20;
unsupervised = true;
supervised = false; 
folderSave='movingDots2';
trajectoryPlot=false;
performancePlot=true;
to_test = true;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, folderSave, trajectoryPlot, performancePlot, to_test);

%% GESTURE NEW

%parpool(50)
dataset = {'inputDatasets/gesture_train','inputDatasets/gesture_test'};
total_columns = 30;
unsupervised = false;
supervised = false; 
folderSave='gesture_taskSwitch';
trajectoryPlot=false;
performancePlot=false;
to_test = false;
task_switch = true;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, folderSave, trajectoryPlot, performancePlot, to_test);

%% 2moving obj small NEW

dataset = {'inputDatasets/2movingObj_small.mat','inputDatasets/2movingObj_small.mat'};
total_columns = 2;
unsupervised = false;
supervised = false; 
folderSave='2movingObj_sm_taskSwitch';
trajectoryPlot=false;
performancePlot=true;
to_test = false;
taskSwitch = true;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, taskSwitch, folderSave, trajectoryPlot, performancePlot, to_test);

%% 2moving obj small
parpool(50)
dataset = {'inputDatasets/2movingObj_small.mat'};
total_columns = 10;
unsupervised = true;
supervised = false; 
folderSave='2movingObj_sm';
trajectoryPlot=false;
performancePlot=false;
to_test = false;
task_switch = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, folderSave, trajectoryPlot, performancePlot, to_test);

%% unsupervise train - moving dots
%parpool(50)
dataset = {'inputDatasets/varyCo_data_test/varyCo8_70.mat', 'inputDatasets/varyCo_data_train/varyCo_70.mat'};
total_columns = 1;
unsupervised = true;
supervised = false; 
folderSave='movingDots_70';
trajectoryPlot=false;
performancePlot=true;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% unsupervise train - moving dots
%parpool(50)
dataset = {'inputDatasets/varyCoh_train_all.mat', 'inputDatasets/varyCo_data_test/varyCo8_10.mat'};%, 'inputDatasets/varyCo_data_train/varyCo_70.mat'};
total_columns = 20;
unsupervised = true;
supervised = false; 
folderSave='movingDots_allCoh';
trajectoryPlot=true;
performancePlot=true;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);


%% supervise train - moving dots
%parpool(50)
dataset = {'inputDatasets/varyCo_data_train/varyCo_70.mat','inputDatasets/varyCo_data_train/varyCo_90.mat'};
total_columns = 20;
unsupervised = true;
supervised = false; 
folderSave='movingDots_train-70_test-90_US_20col_hebb2';
trajectoryPlot=true;
performancePlot=true;
to_test = true;
task_switch = false;
motion_predict = false;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motion_predict, folderSave, trajectoryPlot, performancePlot, to_test);

%% task switch train - moving dots
%parpool(50)
dataset = {'inputDatasets/varyCo_data_train/varyCo_70.mat','inputDatasets/varyCo_data_test/varyCo_70_test.mat'};
total_columns = 20;
unsupervised = false;
supervised = false; 
folderSave='movingDots_70_taskSwitch_20cols';
trajectoryPlot=false;
performancePlot=true;
to_test = true;
task_switch = true;
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, folderSave, trajectoryPlot, performancePlot, to_test);

%% 10 moving obj 

dataset = {'inputDatasets/10movingObj.mat'};
total_columns = 1;
unsupervised = true;
supervised = false; 
folderSave='10movingObj';
trajectoryPlot=false;
performancePlot=false;
to_test = false;
task_switch = false;
motionPredict = false; 
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motionPredict, folderSave, trajectoryPlot, performancePlot, to_test);


%% Moving obj with tests

dataset = {'inputDatasets/snapShot_2objects_train.mat', 'inputDatasets/snapShot_2objects_test_pxl.mat'};
total_columns = 1;
unsupervised = true;
supervised = false; 
folderSave='2MovingObj_test_hebb';
trajectoryPlot=false;
performancePlot=false;
to_test = true;
task_switch = false;
motionPredict = false; 
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motionPredict, folderSave, trajectoryPlot, performancePlot, to_test);

%% Moving obj with motion prediction

dataset = {'inputDatasets/snapShot_2objects_train.mat', 'inputDatasets/snapShot_2objects_test_pxl.mat'};
total_columns = 1;
unsupervised = true;
supervised = false; 
folderSave='2movingObj_samePath';
trajectoryPlot=false;
performancePlot=false;
to_test = true;
task_switch = false;
motionPredict = false; 
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motionPredict, folderSave, trajectoryPlot, performancePlot, to_test);


%% Moving obj with motion prediction

dataset = {'inputDatasets/snapShot_6objects_train.mat', 'inputDatasets/snapShot_6objects_test_pxl.mat'};
total_columns = 1;
unsupervised = true;
supervised = false; 
folderSave='6movingObj_samePath';
trajectoryPlot=false;
performancePlot=false;
to_test = true;
task_switch = false;
motionPredict = false; 
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motionPredict, folderSave, trajectoryPlot, performancePlot, to_test);


%% 14 Moving obj - classification

dataset = {'inputDatasets/snapShot_14objects_train.mat', 'inputDatasets/snapShot_14objects_test_pxl.mat'};
total_columns = 10;
unsupervised = true;
supervised = false; 
folderSave='14MovingObj';
trajectoryPlot=false;
performancePlot=false;
to_test = true;
task_switch = false;
motionPredict = false; 
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motionPredict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 5 Moving obj - 5 classes

dataset = {'inputDatasets/snapShot_5objects_train.mat', 'inputDatasets/snapShot_5objects_test_pxl.mat'};
total_columns = 20;
unsupervised = true;
supervised = false; 
folderSave='5movingobj_5class';

%% 2 Moving obj (varying shape)

dataset = {'inputDatasets/varying_object_shape_2_train.mat'};
total_columns = 1;
unsupervised = true;
supervised = false; 
folderSave='vary3Obj_shape';
trajectoryPlot=false;
performancePlot=false;
to_test = false;
task_switch = false;
motionPredict = false; 
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motionPredict, folderSave, trajectoryPlot, performancePlot, to_test);

%% 3 different directions of Moving obj 
% parfor a = 1:15
dataset = {'inputDatasets/snapShot_9patterns_train', 'inputDatasets/snapShot_9patterns_test_vel-1.mat'};
total_columns = 1;
num_layers = 3
unsupervised = true;
supervised = false; 
folderSave='7_layer_test_metric_paper_2';
trajectoryPlot=true;
performancePlot=true;
to_test = true;
task_switch = false;
motionPredict = false; 
multiColumn_SNN(dataset, total_columns, unsupervised, supervised, task_switch, motionPredict, folderSave, trajectoryPlot, performancePlot, to_test,num_layers);

% end



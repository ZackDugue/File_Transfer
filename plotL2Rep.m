figure(1); 


for tt = 1:200
    firedUnits = find(L2H(:,tt) == 1);
    scatter(L2Pos(:,1), L2Pos(:,2), 'k', 'filled')
    hold on
    scatter(L2Pos(firedUnits,1), L2Pos(firedUnits,2), 'r', 'filled')
    title(sprintf('tt = %d',tt))
    pause(0.1)
    cla()
end
%%
figure(2); 

scatter3(L2Pos(:,1), L2Pos(:,2), L2Pos(:,3), 'k', 'filled')
hold on

for tt = 1:5:400
    firedUnits = find(L2H(:,tt) == 1);
    scatter3(L2Pos(firedUnits,1), L2Pos(firedUnits,2), repmat(tt+1, length(firedUnits),1), 'r', '.')
end

for tt = 401:5:800
    firedUnits = find(L2H(:,tt) == 1);
    scatter3(L2Pos(firedUnits,1), L2Pos(firedUnits,2), repmat(tt+1, length(firedUnits),1), 'g', '.')
end

for tt = 801:5:1200
    firedUnits = find(L2H(:,tt) == 1);
    scatter3(L2Pos(firedUnits,1), L2Pos(firedUnits,2), repmat(tt+1, length(firedUnits),1), 'b', '.')
end

zlabel('Time(t)')

%%
figure(3); 

scatter3(L2Pos(:,1), L2Pos(:,2), L2Pos(:,3), 'k', 'filled')
hold on

for tt = 1:5:300
    z = L2v(:,tt);
    zscaled = z*10;                                                 % May Be Necessary To Scale The Colour Vector
    zscaled = zscaled - min(zscaled)+1;
    cn = ceil(max(zscaled));                                        % Number Of Colors (Scale AsAppropriate)
    cm = colormap(jet(cn));                                         % Define Colormap
    scatter3(L2Pos(:,1), L2Pos(:,2), repmat(tt+1, length(L2Pos(:,1)),1), [], cm(ceil(zscaled),:), 'filled')
    colorbar
    %pause(0.3)
end

for tt = 301:5:600
    firedUnits = find(L2H(:,tt) == 1);
    scatter3(L2Pos(firedUnits,1), L2Pos(firedUnits,2), repmat(tt+1, length(firedUnits),1), 'b', 'filled')
end

%%

figure(4); 
scatter(L2Pos(:,1), L2Pos(:,2), 'k', 'filled')
hold on

L2.Hwin = zeros(size(selfOrgNet(1,2).Hwin)); 

for tt = 1:300
    L2.Hwin = circshift(L2.Hwin,1,2); 
    L2.Hwin(:,1) = L2H(:,tt); 
    L2.Hint = sum(L2.Hwin, 2);   % Top Hat filter  
    firedUnits = find(L2.Hint > 0);
    scatter(L2Pos(:,1), L2Pos(:,2), [], L2.Hint, 'filled')
    colorbar
    pause(0.3)
    %scatter3(L2Pos(firedUnits,1), L2Pos(firedUnits,2), repmat(tt+1, length(firedUnits),1), 'r', 'filled')
end


%% PLOT LOSS VALUE (UNSUPERVISED)

x = [];
g = [];
legendCell = {};
ep_sum = [];

for ii = 1:length(netLossVal)
    
    x = [x, netLossVal{ii}];
    ep_sum = [ep_sum, sum(netLossVal{ii})];
    g = [g; (ii-1)*ones(length(netLossVal{ii}),1)];
    legendCell{ii} = sprintf('Epoch-%d',ii); 
end

figure; 
boxplot(x,g)
xticklabels(legendCell)

ylabel('Unsupervised loss')

figure; 
plot(x)
hold on
% plot(1:length(x), repmat(0.8, length(x),1),'r--')
ylabel('Unsupervised loss')

figure;
plot(3:3:30, ep_sum(3:3:30));
ylabel('Cumulative unsupervised loss')
xlabel('Epochs')

load('9patternsN_hebb2/AE/AE_column=1_ep=3_test.mat')
x2 = netLossVal{3};

figure;
num_patterns = 9;
num_ex_eachPattern= length(x2)/num_patterns;
idx = 1;

figure; 
hold on
for ii = 1:num_patterns
    plot(idx:idx+num_ex_eachPattern-1, x2(idx:idx+num_ex_eachPattern-1))
    idx = idx + num_ex_eachPattern;
end
xlabel('Time(t)')
ylabel('Unsupervised loss')


load('3movingObj_nonHebb/AE/AE_column=1_ep=5_test.mat')
x2 = netLossVal{5};
%figure;
plot(1:80, x2(1:80),'r')
hold on
plot(81:160, x2(81:160),'g')
plot(161:240, x2(161:240),'b')
xlabel('Time(t)')
ylabel('Unsupervised loss')

%% HEBB vs NON_HEBB 

figure; 
load('9patternsN_hebb2/AE/AE_column=10_ep=3_test.mat')
x2 = netLossVal{3};
plot(x2, 'b')
hold on
load('9patternsN_nonhebb/AE/AE_column=1_ep=3_test.mat')
x2 = netLossVal{1};
plot(x2, 'r')
legend('Hebb', 'Non-Hebb')
xlabel('Time(t)')
ylabel('Unsupervised loss')



figure;
load('9patternsN_hebb2/AE/AE_column=10_ep=3_test.mat')
ep_sum = [];
for ii = 1:length(netLossVal)
    ep_sum = [ep_sum, sum(netLossVal{ii})];
end

plot(ep_sum, 'b')
hold on
load('9patternsN_nonhebb/AE/AE_column=1_ep=3_test.mat')
ep_sum = [];
for ii = 1:length(netLossVal)
    ep_sum = [ep_sum, sum(netLossVal{ii})];
end

plot(ep_sum, 'r')
legend('Hebb', 'Non-Hebb')
ylabel('Cumulative unsupervised loss')
xlabel('Epochs')


%% Plot RealTime accuracy (for last epoch)

RT_train = RT_trainMat{1}{1};
t_idx = 1:size(RT_train,2);

[mVal, mInd] = max(RT_train, [],1);
keepIdx = find(mVal ~= 0);
mInd = mInd(keepIdx);
t_idx = t_idx(keepIdx);

figure; 
plot(t_idx, mInd)
hold on
t_idx = 1:size(RT_train,2);
u = repelem(labels_all,10);
plot(t_idx, u, 'r--')


legend('Real-time predicted label', 'True label')
xlabel('Time(t)')
ylabel('Moving object label')


RT_test = RT_testMat{1}{1};
t_idx = 1:size(RT_test,2);

[mVal, mInd] = max(RT_test, [],1);
keepIdx = find(mVal ~= 0);
mInd = mInd(keepIdx);
t_idx = t_idx(keepIdx);

figure; 
plot(t_idx, mInd)
hold on
t_idx = 1:size(RT_test,2);
u = repelem(labels_all,10);
plot(t_idx, u, 'r--')


legend('Real-time predicted label', 'True label')
xlabel('Time(t)')
ylabel('Moving object label')

load('9patternsN_hebb2/DiscrimPerf.mat')
figure; 
tmp = [acc_train{10}{:}];
plot(tmp(1:2:end-4))
xlabel('Epochs')
%ylim([min(tmp(1:2:end)) 100])
hold on
tmp = [acc_test{10}{:}];
plot(tmp(1:2:end-4))
xlabel('Epochs')
ylabel({['Averaged Performance'], ['(over entire duration)']})
ylim([min(tmp(1:2:end)) 100])
legend('Train', 'Test')


%% HEBB vs NON-HEBB
load('9patternsN_nonhebb/DiscrimPerf.mat')
tmp = [acc_train{1}{:}];
plot(tmp(1:2:end-4),'b')
hold on
tmp = [acc_test{1}{:}];
plot(tmp(1:2:end-4),'b--')

load('9patternsN_hebb2/DiscrimPerf.mat')
tmp = [acc_train{10}{:}];
plot(tmp(1:2:end-4),'r')
hold on
tmp = [acc_test{10}{:}];
plot(tmp(1:2:end-4),'r--')

xlabel('Epochs')
ylabel('Performance')

legend('Non-Hebb Train', 'Non-Hebb Test', 'Hebb-Train', 'Hebb-Test')

% for tt = 1:5:300
%     firedUnits = find(L2H(:,tt) == 1);
%     scatter3(L2Pos(firedUnits,1), L2Pos(firedUnits,2), repmat(tt+1, length(firedUnits),1), 'r', 'filled')
% end
% 
% for tt = 301:5:600
%     firedUnits = find(L2H(:,tt) == 1);
%     scatter3(L2Pos(firedUnits,1), L2Pos(firedUnits,2), repmat(tt+1, length(firedUnits),1), 'b', 'filled')
% end
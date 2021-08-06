%% CREATING MOVING OBJECTS ALONG DIFFERENT DIRECTIONS

clear; clc;
close all

% System Constants and Parameters
sqR1 = 80;   sqR2 = 101; 
nR = sqR1*sqR2;      % # Neurons in Retina

% Retina Structure Parameters
Ret = {};       % Retina Data Structure
Ret.th = ones(nR,1);        %variable retina thresh
Ret.v_reset = 0 + 0.1*randn(nR,1).^2;     %Noise on activity field
Ret.v = 0*ones(nR,1); %I.C. of v

% Ret.nx = meshgrid([0.5:1:sqR2-0.5],[0.5:1:sqR1-0.5]) + unifrnd(-0.5,0.5,sqR1,sqR2);
[X,Y] = meshgrid([0.5:1:sqR2-0.5],[sqR1-0.5:-1:0.5]); 
Ret.nx = [ X(:) Y(:) ];

Ret.H = sparse(zeros(nR,1)); % equivalent to "spikeMat"
Ret.eta = []; 
Ret.htmp = zeros(nR,1); % Heatmap # of times each neuron spikes
label_all = [];


num_obj = input('Number of objects = ');
num_traj = input('Number of trajectories = ');

trajectory = cell(num_traj,1);

figure;
for trajNum = 1:num_traj
    
    scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
    hold on
    
    [myobj,xs,ys] = freehanddraw(gca,'color','r','linewidth',3);
    
    trajectory{trajNum} = [xs,ys];
end

obj = cell(num_obj,1);
for objNum = 1:num_obj
    
    figure; 
    scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
    hold on
    
    % Get object position/dimensions
    h = imfreehand;
    pos_obj = h.getPosition();
    [in, on] = inpolygon(Ret.nx(:,1), Ret.nx(:,2), pos_obj(:,1), pos_obj(:,2));

    obj{objNum}.pos = Ret.nx(find(in==1),:);
    
%     % Get object centroid
%     centroid = ginput(1);
    obj{objNum}.centroid = mean(obj{objNum}.pos);
    obj{objNum}.startPos = obj{objNum}.pos;    
   
    % get start time of object motion
    obj{objNum}.stTime = input('start time: ');
    obj{objNum}.endTime = input('end time: ');
end

obj2 = cell(1, num_obj * num_traj);

ctr = 1;

for objNum = 1:num_obj
    %trajNum = objNum;
    for trajNum = 1:num_traj
        
        [X_traj] = trajectory{trajNum};
        
        disp = X_traj(1,:) - obj{objNum}.centroid;
        obj2{ctr}.pos = obj{objNum}.pos + disp;
        
        % find closest ret pixels
        D = pdist2(Ret.nx, obj2{ctr}.pos);
        [minVal, minInd] = min(D, [], 1);
        obj2{ctr}.pos = Ret.nx(minInd,:);
        
        obj2{ctr}.centroid = mean(obj2{ctr}.pos);
        obj2{ctr}.startPos = obj2{ctr}.pos;
        
        obj2{ctr}.stTime = obj{objNum}.stTime;
        obj2{ctr}.endTime = obj{objNum}.endTime;
        obj2{ctr}.prevVal = 1;
        obj2{ctr}.traj = X_traj;
        ctr = ctr + 1;
        
    end
end


% SIMULATE object motion over time 

num_patterns = num_obj * num_traj;

objTime = obj2{1}.endTime - obj2{1}.stTime;
tt = 1;
for objNum = 1:num_patterns
    
    obj2{objNum}.stTime = tt;
    obj2{objNum}.endTime = tt+objTime;
    tt = obj2{objNum}.endTime+1 ;
end
    

totTime = 0;
for objNum = 1:num_patterns
    totTime = max(totTime, obj2{objNum}.endTime);
end


snapShotMat = zeros(size(Ret.nx,1),totTime);
labels_all = zeros(num_patterns, totTime);

for objNum = 1:num_patterns
    labels_all(objNum, obj2{objNum}.stTime:obj2{objNum}.endTime) = objNum;
    speed = 2; %1+rand(1);
    %vel = obj{objNum}.endPos - obj{objNum}.centroid;
    %vel = vel/sqrt(sum(vel.^2));
    obj2{objNum}.speed = speed; 
    obj2{objNum}.histPos = {};
end

% figure(1);
% scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
% hold on
% scatter(obj2{1}.traj(:,1),obj2{1}.traj(:,2),'r', 'filled')
% scatter(obj2{1}.pos(:,1),obj2{1}.pos(:,2),'b', 'filled')
% scatter(obj2{1}.centroid(:,1),obj2{1}.centroid(:,2),'g', 'filled')
% cla()


pos_obj_train = {};
for tt = 1:totTime
    ind_all = [];
        
    for objNum = 1:num_patterns
        
        
        if and(tt>=obj2{objNum}.stTime, tt<=obj2{objNum}.endTime)
            
            %vel = obj{objNum}.endPos - obj{objNum}.centroid;
            %vel = vel/sqrt(sum(vel.^2));
            D = pdist2(obj2{objNum}.traj, obj2{objNum}.centroid);
            [minVal, minInd] = min(D);
            %minInd, obj2{objNum}.prevVal
            if minInd <= obj2{objNum}.prevVal
                minInd = obj2{objNum}.prevVal+1;
            end
            
            if minInd >= length(D)-1
                % reset at start position
                obj2{objNum}.pos = obj2{objNum}.startPos;
                obj2{objNum}.centroid = mean(obj2{objNum}.pos);                
                minInd = 1;
                obj2{objNum}.prevVal = 1;
            end
            
            dir = obj2{objNum}.traj(minInd+1,:) - obj2{objNum}.centroid;
            vel = obj2{objNum}.speed * dir/sqrt(sum(dir.^2));
            obj2{objNum}.prevVal = minInd;
            
            obj2{objNum}.pos = obj2{objNum}.pos + vel;
            obj2{objNum}.centroid = obj2{objNum}.centroid + vel;

            
%             figure(1);
%             scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
%             hold on
%             scatter(obj2{1}.traj(:,1),obj2{1}.traj(:,2),'r', 'filled')
%             scatter(obj2{1}.pos(:,1),obj2{1}.pos(:,2),'b', 'filled')
%             scatter(obj2{1}.centroid(:,1),obj2{1}.centroid(:,2),'g', 'filled')
            
            
            objPos_pixels = size(obj2{objNum}.pos,1);
            % Find if object pos is outside boundary or not
            outlier_x = union(find(obj2{objNum}.pos(:,1) < min(Ret.nx(:,1))), find(obj2{objNum}.pos(:,1) > max(Ret.nx(:,1))));
            outlier_y = union(find(obj2{objNum}.pos(:,2) < min(Ret.nx(:,2))), find(obj2{objNum}.pos(:,2) > max(Ret.nx(:,2))));
            
            pix_out = length(outlier_x) + length(outlier_y);
            if pix_out > 0.2*objPos_pixels
                obj2{objNum}.pos = obj2{objNum}.startPos;
                %obj2{objNum}.centroid = mean(obj2{objNum}.pos);  
            end
            
%             %Check if its within the boundary or not
%             if or(obj{objNum}.centroid(1) < min(Ret.nx(:,1)), obj{objNum}.centroid(1) > max(Ret.nx(:,1)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
% 
%             if or(obj{objNum}.centroid(2) < min(Ret.nx(:,2)), obj{objNum}.centroid(2) > max(Ret.nx(:,2)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
            
            dist_mat_pos = pdist2(obj2{objNum}.pos, Ret.nx);
            dist_mat_centroid = pdist2(obj2{objNum}.centroid, Ret.nx);

            [~,ind] = min(dist_mat_pos,[],2);
            obj2{objNum}.pos = Ret.nx(ind,:);
            ind_all = [ind_all; ind];
            
            %pos_obj_train{end+1} = obj2{objNum}.pos;
            obj2{objNum}.histPos{end+1} = obj2{objNum}.pos;
            
            %[~,ind] = min(dist_mat_centroid,[],2);
            obj2{objNum}.centroid = mean(obj2{objNum}.pos);
            
        end
        
    end
    RetSpikeStatus = zeros(size(Ret.nx,1),1);
    RetSpikeStatus(ind_all) = 1;

    snapShotMat(:,tt) = RetSpikeStatus;
    
    figure(1); 
    scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
    hold on
    scatter(Ret.nx(find(RetSpikeStatus==1),1),Ret.nx(find(RetSpikeStatus==1),2),'r','filled')
    for objNum = 1:num_patterns
        scatter(obj2{objNum}.traj(:,1), obj2{objNum}.traj(:,2), 'b.')
        hold on
    end
        
    pause(0.1)
    
end
data_all = snapShotMat;
labels_all = sum(labels_all);

labels2 = []; 
for ii = 1:length(labels_all)
    labels2 = [labels2, find(labels_all(ii) == unique(labels_all))];
end
labels_all = labels2;
save(sprintf('snapShot_%dpatternsN_train.mat', num_patterns),'data_all','labels_all','obj')


%% TEST DATASET

for vel_idx = 1:1
% 
% obj2 = cell(1, num_obj * num_traj);
% 
% ctr = 1;
% 
% for objNum = 1:num_obj
%     %trajNum = objNum;
%     
%     for trajNum = 1:num_traj
%         
%         [X_traj] = trajectory{trajNum};
%         
%         disp = X_traj(1,:) - obj{objNum}.centroid;
%         obj2{ctr}.pos = obj{objNum}.pos + disp;
%         
%         % find closest ret pixels
%         D = pdist2(Ret.nx, obj2{ctr}.pos);
%         [minVal, minInd] = min(D, [], 1);
%         obj2{ctr}.pos = Ret.nx(minInd,:);
%         
%         obj2{ctr}.centroid = mean(obj2{ctr}.pos);
%         obj2{ctr}.startPos = obj2{ctr}.pos;
%         
%         obj2{ctr}.stTime = obj{objNum}.stTime;
%         obj2{ctr}.endTime = obj{objNum}.endTime;
%         obj2{ctr}.prevVal = 1;
%         obj2{ctr}.traj = X_traj;
%         ctr = ctr + 1;
%         
%     end
% end
% 
% 
% % SIMULATE object motion over time 
% 
% num_patterns = num_obj* num_traj;
% 
% objTime = obj2{1}.endTime - obj2{1}.stTime;
% tt = 1;
% for objNum = 1:num_patterns
%     
%     obj2{objNum}.stTime = tt;
%     obj2{objNum}.endTime = tt+objTime;
%     tt = obj2{objNum}.endTime+1 ;
% end
%     
% 
% totTime = 0;
% for objNum = 1:num_patterns
%     totTime = max(totTime, obj2{objNum}.endTime);
% end
% 

snapShotMat = zeros(size(Ret.nx,1),totTime);

% for objNum = 1:num_patterns
%     labels_all(objNum, obj2{objNum}.stTime:obj2{objNum}.endTime) = objNum;
%     speed = 5; %1+1*rand(1);
%     %vel = obj{objNum}.endPos - obj{objNum}.centroid;
%     %vel = vel/sqrt(sum(vel.^2));
%     obj2{objNum}.speed = speed;    
% end

% figure(1);
% scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
% hold on
% scatter(obj2{1}.traj(:,1),obj2{1}.traj(:,2),'r', 'filled')
% scatter(obj2{1}.pos(:,1),obj2{1}.pos(:,2),'b', 'filled')
% scatter(obj2{1}.centroid(:,1),obj2{1}.centroid(:,2),'g', 'filled')
% cla()

ctr = 1;
for tt = 1:totTime
    ind_all = [];
        
    for objNum = 1:num_patterns
        
        if and(tt>=obj2{objNum}.stTime, tt<=obj2{objNum}.endTime)
            
            %vel = obj{objNum}.endPos - obj{objNum}.centroid;
            %vel = vel/sqrt(sum(vel.^2));
            D = pdist2(obj2{objNum}.traj, obj2{objNum}.centroid);
            [minVal, minInd] = min(D);
            %minInd, obj2{objNum}.prevVal
            if minInd < obj2{objNum}.prevVal
                minInd = obj2{objNum}.prevVal+1;
            end
            
            if minInd >= length(D)-1
                % reset at start position
                obj2{objNum}.pos = obj2{objNum}.startPos;
                obj2{objNum}.centroid = mean(obj2{objNum}.pos);                
                minInd = 1;
                obj2{objNum}.prevVal = 1;
            end
            
            %dir = obj2{objNum}.traj(minInd+1,:) - obj2{objNum}.centroid + 2*randn(1);
            %vel = obj2{objNum}.speed * dir/sqrt(sum(dir.^2));
            obj2{objNum}.prevVal = minInd;
            
            %obj2{objNum}.pos = obj2{objNum}.pos + vel;
            %obj2{objNum}.centroid = obj2{objNum}.centroid + vel;

            obj2{objNum}.pos = obj2{objNum}.histPos{tt-obj2{objNum}.stTime+1} + 2*randn(size(mean(obj2{objNum}.pos)));
            obj2{objNum}.centroid = mean(obj2{objNum}.pos);
            
            
%             figure(1);
%             scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
%             hold on
%             scatter(obj2{1}.traj(:,1),obj2{1}.traj(:,2),'r', 'filled')
%             scatter(obj2{1}.pos(:,1),obj2{1}.pos(:,2),'b', 'filled')
%             scatter(obj2{1}.centroid(:,1),obj2{1}.centroid(:,2),'g', 'filled')
            
            
            objPos_pixels = size(obj2{objNum}.pos,1);
            % Find if object pos is outside boundary or not
            outlier_x = union(find(obj2{objNum}.pos(:,1) < min(Ret.nx(:,1))), find(obj2{objNum}.pos(:,1) > max(Ret.nx(:,1))));
            outlier_y = union(find(obj2{objNum}.pos(:,2) < min(Ret.nx(:,2))), find(obj2{objNum}.pos(:,2) > max(Ret.nx(:,2))));
            
            pix_out = length(outlier_x) + length(outlier_y);
            if pix_out > 0.2*objPos_pixels
                obj2{objNum}.pos = obj2{objNum}.startPos;
                %obj2{objNum}.centroid = mean(obj2{objNum}.pos);  
            end
            
%             %Check if its within the boundary or not
%             if or(obj{objNum}.centroid(1) < min(Ret.nx(:,1)), obj{objNum}.centroid(1) > max(Ret.nx(:,1)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
% 
%             if or(obj{objNum}.centroid(2) < min(Ret.nx(:,2)), obj{objNum}.centroid(2) > max(Ret.nx(:,2)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
            
            dist_mat_pos = pdist2(obj2{objNum}.pos, Ret.nx);
            dist_mat_centroid = pdist2(obj2{objNum}.centroid, Ret.nx);

            [~,ind] = min(dist_mat_pos,[],2);
            obj2{objNum}.pos = Ret.nx(ind,:);
            ind_all = [ind_all; ind];
            
            %[~,ind] = min(dist_mat_centroid,[],2);
            obj2{objNum}.centroid = mean(obj2{objNum}.pos);
            
        end
        ctr = ctr + 1;
        
    end
    RetSpikeStatus = zeros(size(Ret.nx,1),1);
    RetSpikeStatus(ind_all) = 1;

    snapShotMat(:,tt) = RetSpikeStatus;
    
%     figure(1); 
%     scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
%     hold on
%     scatter(Ret.nx(find(RetSpikeStatus==1),1),Ret.nx(find(RetSpikeStatus==1),2),'r','filled')
%     for objNum = 1:num_patterns
%         scatter(obj2{objNum}.traj(:,1), obj2{objNum}.traj(:,2), 'b.')
%         hold on
%     end
%         
%     pause(0.1)
    
end
data_all = snapShotMat;


save(sprintf('snapShot_%dpatternsN_test-%d.mat', num_patterns, vel_idx),'data_all','labels_all','obj')

end
%% TEST DATASET
% 
% snapShotMat = zeros(size(Ret.nx,1),totTime);
% labels_all = zeros(num_obj, totTime);
% 
% for objNum = 1:num_obj
%     labels_all(objNum, obj{objNum}.stTime:obj{objNum}.endTime) = objNum;
%     
%     obj{objNum}.pos = obj{objNum}.startPos;
%     obj{objNum}.centroid = mean(obj{objNum}.pos);
%     
%     vel = obj{objNum}.endPos - obj{objNum}.centroid;
%     vel = vel/sqrt(sum(vel.^2));
%     obj{objNum}.vel = vel;
%     
%     pxl_flicker{objNum} = datasample(1:size(obj{objNum}.pos,1),floor(size(obj{objNum}.pos,1)/1.8),'Replace',false);
%     
% end
% 
% 
% 
% for tt = 1:totTime
%     ind_all = [];
%     for objNum = 1:num_obj
%         
%         if and(tt>=obj{objNum}.stTime, tt<=obj{objNum}.endTime)
%             
%             %vel = obj{objNum}.endPos - obj{objNum}.centroid;
%             %vel = vel/sqrt(sum(vel.^2));
%             obj{objNum}.pos = obj{objNum}.pos + obj{objNum}.vel;
%             obj{objNum}.centroid = obj{objNum}.centroid + obj{objNum}.vel;
%             
%             
% 
%             objPos_pixels = size(obj{objNum}.pos,1);
%             % Find if object pos is outside boundary or not
%             outlier_x = union(find(obj{objNum}.pos(:,1) < min(Ret.nx(:,1))), find(obj{objNum}.pos(:,1) > max(Ret.nx(:,1))));
%             outlier_y = union(find(obj{objNum}.pos(:,2) < min(Ret.nx(:,2))), find(obj{objNum}.pos(:,2) > max(Ret.nx(:,2))));
%             
%             pix_out = length(outlier_x) + length(outlier_y);
%             if pix_out > 0.2*objPos_pixels
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
%             
% %             %Check if its within the boundary or not
% %             if or(obj{objNum}.centroid(1) < min(Ret.nx(:,1)), obj{objNum}.centroid(1) > max(Ret.nx(:,1)))
% %                 obj{objNum}.pos = obj{objNum}.startPos;
% %             end
% % 
% %             if or(obj{objNum}.centroid(2) < min(Ret.nx(:,2)), obj{objNum}.centroid(2) > max(Ret.nx(:,2)))
% %                 obj{objNum}.pos = obj{objNum}.startPos;
% %             end
%             
%             dist_mat_pos = pdist2(obj{objNum}.pos, Ret.nx);
%             dist_mat_centroid = pdist2(obj{objNum}.centroid, Ret.nx);
% 
%             [~,ind] = min(dist_mat_pos,[],2);
%             obj{objNum}.pos = Ret.nx(ind,:);
%             
%             %ind(pxl_flicker{objNum}) = [];
%             ind(datasample(1:size(obj{objNum}.pos,1),floor(size(obj{objNum}.pos,1)/1.8),'Replace',false)) = [];
%             
%             ind_all = [ind_all; ind];
%             
%             %[~,ind] = min(dist_mat_centroid,[],2);
%             obj{objNum}.centroid = mean(obj{objNum}.pos);
%             
%         end
%         
%     end
%     RetSpikeStatus = zeros(size(Ret.nx,1),1);
%     RetSpikeStatus(ind_all) = 1;
% 
%     snapShotMat(:,tt) = RetSpikeStatus;
%     
%     figure(1); 
%     scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
%     hold on
%     scatter(Ret.nx(find(RetSpikeStatus==1),1),Ret.nx(find(RetSpikeStatus==1),2),'r','filled')
%     pause(0.1)
% 
%     
% end
% data_all = snapShotMat;
% labels_all = sum(labels_all);
% 
% labels2 = []; 
% for ii = 1:length(labels_all)
%     labels2 = [labels2, find(labels_all(ii) == unique(labels_all))];
% end
% labels_all = labels2;
% 
% save(sprintf('snapShot_%dobjects_test_pxl.mat',objNum),'data_all','labels_all','obj')

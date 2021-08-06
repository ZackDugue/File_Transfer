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


num_obj = input('num_objects = ');
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
    
    % Get final position of object
    obj{objNum}.endPos = ginput(1);
    
    % get start time of object motion
    obj{objNum}.stTime = input('start time: ');
    obj{objNum}.endTime = input('end time: ');
    
    %scatter(obj{objNum}.pos(:,1), obj{objNum}.pos(:,2),'r','filled')
    %obj{objNum}.vel_all = 2*rand(150,2);

end

% SIMULATE object motion over time 

totTime = 0;
for objNum = 1:num_obj
    totTime = max(totTime, obj{objNum}.endTime);
end

snapShotMat = zeros(size(Ret.nx,1),totTime);
labels_all = zeros(num_obj, totTime);

for objNum = 1:num_obj
    labels_all(objNum, obj{objNum}.stTime:obj{objNum}.endTime) = objNum;
    vel = obj{objNum}.endPos - obj{objNum}.centroid;
    vel = vel/sqrt(sum(vel.^2));
    obj{objNum}.vel = vel;    
end


for tt = 1:totTime
    ind_all = [];
    for objNum = 1:num_obj
        
        if and(tt>=obj{objNum}.stTime, tt<=obj{objNum}.endTime)
            
            %vel = obj{objNum}.endPos - obj{objNum}.centroid;
            %vel = vel/sqrt(sum(vel.^2));
            obj{objNum}.pos = obj{objNum}.pos + obj{objNum}.vel;
            obj{objNum}.centroid = obj{objNum}.centroid + obj{objNum}.vel;
            
            objPos_pixels = size(obj{objNum}.pos,1);
            % Find if object pos is outside boundary or not
            outlier_x = union(find(obj{objNum}.pos(:,1) < min(Ret.nx(:,1))), find(obj{objNum}.pos(:,1) > max(Ret.nx(:,1))));
            outlier_y = union(find(obj{objNum}.pos(:,2) < min(Ret.nx(:,2))), find(obj{objNum}.pos(:,2) > max(Ret.nx(:,2))));
            
            pix_out = length(outlier_x) + length(outlier_y);
            if pix_out > 0.2*objPos_pixels
                obj{objNum}.pos = obj{objNum}.startPos;
            end
            
%             %Check if its within the boundary or not
%             if or(obj{objNum}.centroid(1) < min(Ret.nx(:,1)), obj{objNum}.centroid(1) > max(Ret.nx(:,1)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
% 
%             if or(obj{objNum}.centroid(2) < min(Ret.nx(:,2)), obj{objNum}.centroid(2) > max(Ret.nx(:,2)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
            
            dist_mat_pos = pdist2(obj{objNum}.pos, Ret.nx);
            dist_mat_centroid = pdist2(obj{objNum}.centroid, Ret.nx);

            [~,ind] = min(dist_mat_pos,[],2);
            obj{objNum}.pos = Ret.nx(ind,:);
            ind_all = [ind_all; ind];
            
            %[~,ind] = min(dist_mat_centroid,[],2);
            obj{objNum}.centroid = mean(obj{objNum}.pos);
            
        end
        
    end
    RetSpikeStatus = zeros(size(Ret.nx,1),1);
    RetSpikeStatus(ind_all) = 1;

    snapShotMat(:,tt) = RetSpikeStatus;
    
    figure(1); 
    scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
    hold on
    scatter(Ret.nx(find(RetSpikeStatus==1),1),Ret.nx(find(RetSpikeStatus==1),2),'r','filled')
    pause(0.1)

    
end
data_all = snapShotMat;
labels_all = sum(labels_all);

labels2 = []; 
for ii = 1:length(labels_all)
    labels2 = [labels2, find(labels_all(ii) == unique(labels_all))];
end
labels_all = labels2;
save(sprintf('snapShot_%dobjects_train.mat',objNum),'data_all','labels_all','obj')


%% TEST DATASET

snapShotMat = zeros(size(Ret.nx,1),totTime);
labels_all = zeros(num_obj, totTime);

for objNum = 1:num_obj
    labels_all(objNum, obj{objNum}.stTime:obj{objNum}.endTime) = objNum;
    
    obj{objNum}.pos = obj{objNum}.startPos;
    obj{objNum}.centroid = mean(obj{objNum}.pos);
    
    vel = obj{objNum}.endPos - obj{objNum}.centroid;
    vel = vel/sqrt(sum(vel.^2))*2;
    obj{objNum}.vel = vel;
    
end


for tt = 1:totTime
    ind_all = [];
    for objNum = 1:num_obj
        
        if and(tt>=obj{objNum}.stTime, tt<=obj{objNum}.endTime)
            
            %vel = obj{objNum}.endPos - obj{objNum}.centroid;
            %vel = vel/sqrt(sum(vel.^2));
            obj{objNum}.pos = obj{objNum}.pos + obj{objNum}.vel;
            obj{objNum}.centroid = obj{objNum}.centroid + obj{objNum}.vel;

            objPos_pixels = size(obj{objNum}.pos,1);
            % Find if object pos is outside boundary or not
            outlier_x = union(find(obj{objNum}.pos(:,1) < min(Ret.nx(:,1))), find(obj{objNum}.pos(:,1) > max(Ret.nx(:,1))));
            outlier_y = union(find(obj{objNum}.pos(:,2) < min(Ret.nx(:,2))), find(obj{objNum}.pos(:,2) > max(Ret.nx(:,2))));
            
            pix_out = length(outlier_x) + length(outlier_y);
            if pix_out > 0.2*objPos_pixels
                obj{objNum}.pos = obj{objNum}.startPos;
            end
            
%             %Check if its within the boundary or not
%             if or(obj{objNum}.centroid(1) < min(Ret.nx(:,1)), obj{objNum}.centroid(1) > max(Ret.nx(:,1)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
% 
%             if or(obj{objNum}.centroid(2) < min(Ret.nx(:,2)), obj{objNum}.centroid(2) > max(Ret.nx(:,2)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
            
            dist_mat_pos = pdist2(obj{objNum}.pos, Ret.nx);
            dist_mat_centroid = pdist2(obj{objNum}.centroid, Ret.nx);

            [~,ind] = min(dist_mat_pos,[],2);
            obj{objNum}.pos = Ret.nx(ind,:);
            ind_all = [ind_all; ind];
            
            %[~,ind] = min(dist_mat_centroid,[],2);
            obj{objNum}.centroid = mean(obj{objNum}.pos);
            
        end
        
    end
    RetSpikeStatus = zeros(size(Ret.nx,1),1);
    RetSpikeStatus(ind_all) = 1;

    snapShotMat(:,tt) = RetSpikeStatus;
    
    figure(1); 
    scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
    hold on
    scatter(Ret.nx(find(RetSpikeStatus==1),1),Ret.nx(find(RetSpikeStatus==1),2),'r','filled')
    pause(0.1)

    
end
data_all = snapShotMat;
labels_all = sum(labels_all);

labels2 = []; 
for ii = 1:length(labels_all)
    labels2 = [labels2, find(labels_all(ii) == unique(labels_all))];
end
labels_all = labels2;

save(sprintf('snapShot_%dobjects_test_vel.mat',objNum),'data_all','labels_all','obj')

%% TEST DATASET

snapShotMat = zeros(size(Ret.nx,1),totTime);
labels_all = zeros(num_obj, totTime);

for objNum = 1:num_obj
    labels_all(objNum, obj{objNum}.stTime:obj{objNum}.endTime) = objNum;
    
    obj{objNum}.pos = obj{objNum}.startPos;
    obj{objNum}.centroid = mean(obj{objNum}.pos);
    
    vel = obj{objNum}.endPos - obj{objNum}.centroid;
    vel = vel/sqrt(sum(vel.^2));
    obj{objNum}.vel = vel;
    
    pxl_flicker{objNum} = datasample(1:size(obj{objNum}.pos,1),floor(size(obj{objNum}.pos,1)/1.8),'Replace',false);
    
end



for tt = 1:totTime
    ind_all = [];
    for objNum = 1:num_obj
        
        if and(tt>=obj{objNum}.stTime, tt<=obj{objNum}.endTime)
            
            %vel = obj{objNum}.endPos - obj{objNum}.centroid;
            %vel = vel/sqrt(sum(vel.^2));
            obj{objNum}.pos = obj{objNum}.pos + obj{objNum}.vel;
            obj{objNum}.centroid = obj{objNum}.centroid + obj{objNum}.vel;
            
            

            objPos_pixels = size(obj{objNum}.pos,1);
            % Find if object pos is outside boundary or not
            outlier_x = union(find(obj{objNum}.pos(:,1) < min(Ret.nx(:,1))), find(obj{objNum}.pos(:,1) > max(Ret.nx(:,1))));
            outlier_y = union(find(obj{objNum}.pos(:,2) < min(Ret.nx(:,2))), find(obj{objNum}.pos(:,2) > max(Ret.nx(:,2))));
            
            pix_out = length(outlier_x) + length(outlier_y);
            if pix_out > 0.2*objPos_pixels
                obj{objNum}.pos = obj{objNum}.startPos;
            end
            
%             %Check if its within the boundary or not
%             if or(obj{objNum}.centroid(1) < min(Ret.nx(:,1)), obj{objNum}.centroid(1) > max(Ret.nx(:,1)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
% 
%             if or(obj{objNum}.centroid(2) < min(Ret.nx(:,2)), obj{objNum}.centroid(2) > max(Ret.nx(:,2)))
%                 obj{objNum}.pos = obj{objNum}.startPos;
%             end
            
            dist_mat_pos = pdist2(obj{objNum}.pos, Ret.nx);
            dist_mat_centroid = pdist2(obj{objNum}.centroid, Ret.nx);

            [~,ind] = min(dist_mat_pos,[],2);
            obj{objNum}.pos = Ret.nx(ind,:);
            
            %ind(pxl_flicker{objNum}) = [];
            ind(datasample(1:size(obj{objNum}.pos,1),floor(size(obj{objNum}.pos,1)/1.8),'Replace',false)) = [];
            
            ind_all = [ind_all; ind];
            
            %[~,ind] = min(dist_mat_centroid,[],2);
            obj{objNum}.centroid = mean(obj{objNum}.pos);
            
        end
        
    end
    RetSpikeStatus = zeros(size(Ret.nx,1),1);
    RetSpikeStatus(ind_all) = 1;

    snapShotMat(:,tt) = RetSpikeStatus;
    
    figure(1); 
    scatter(Ret.nx(:,1), Ret.nx(:,2),'k','filled')
    hold on
    scatter(Ret.nx(find(RetSpikeStatus==1),1),Ret.nx(find(RetSpikeStatus==1),2),'r','filled')
    pause(0.1)

    
end
data_all = snapShotMat;
labels_all = sum(labels_all);

labels2 = []; 
for ii = 1:length(labels_all)
    labels2 = [labels2, find(labels_all(ii) == unique(labels_all))];
end
labels_all = labels2;

save(sprintf('snapShot_%dobjects_test_pxl.mat',objNum),'data_all','labels_all','obj')

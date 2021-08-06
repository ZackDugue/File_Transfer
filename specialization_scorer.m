function spec_score = specialization_scorer(p_kMat)
%extracting the variable from the input
p_k_mat = p_kMat; 

%Creating an array of size (number of columns x number of epochs)
spec_score = zeros(numel(p_k_mat)-1,numel(p_k_mat{1}));

%Iterate through the columns
for col = 1:(numel(p_k_mat)-1)
    disp('col')
    disp(col)
    %Iterate through the epochs
    for epoch = 1:numel(p_k_mat{1})
        disp('epoch')
        disp(epoch)
        
        %Get to p_k (an array of the performance of the network)
        column = p_k_mat{col}
        p_k = column{epoch}
 
        
           
            %class_acc is a vector of size where each element is the rate
            % that the column correctly classifies a class. 
            class_acc = diag(p_k);
            
            %convert class_acc to a unit vector
            unit_class_acc = class_acc / norm(class_acc);
            
            %Find the highest accuracy the column gets on a class 
            [maximum_val, maximum_index] = max(unit_class_acc);
            
            %initialize spec_score_2 as the 
            spec_score_2 = maximum_val;
            
            % iterate through all the classes of class_acc
            for ii = 1:numel(unit_class_acc);
                if ii ~= maximum_index
                spec_score_2 = spec_score_2 - unit_class_acc(ii)/(numel(unit_class_acc)-1);
                end
            end
            
            spec_score(col,epoch) = spec_score_2;
             
end
disp('final spec_score from spec_scorer')
disp(spec_score)
end


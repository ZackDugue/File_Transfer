total_columns = 5;

figure; 
hold on

x = [];
g = [];
legendCell = {};

column_interest = 1:total_columns+1;%[1,2,3,4,5,41];%1:31;

for jj = 1:numel(column_interest)
    
    ii = column_interest(jj);
    tmp = [acc_test{ii}{:}];
    
    x = [x, tmp(1:2:end)];
    g = [g; (ii-1)*ones(length(tmp(2:2:end)),1)];
    
    if ii<=total_columns
        legendCell{jj} = sprintf('C-%d',ii);        
    else
        legendCell{jj} = sprintf('MC');        
    end
        
    %boxplot(tmp(1:2:end))
    
end
boxplot(x,g)
xticklabels(legendCell)


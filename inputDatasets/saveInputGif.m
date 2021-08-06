function [] = saveInputGif(fname)

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

fname_load = strcat(fname,'.mat');
fname_gif = strcat(fname, '.gif');

load(fname_load)
label_all = labels_all;

%%
% h = figure('visible','off');
h = figure;
for ii = 1:2:size(data_all,2)
    
    spikeUnits = find(data_all(:,ii)==1);
    
    scatter(Ret.nx(:,1),Ret.nx(:,2),'k','filled')
    hold on
    scatter(Ret.nx(spikeUnits,1),Ret.nx(spikeUnits,2),'r','filled')
    title(sprintf('Label = %d',label_all(ii)))
    axis off
    pause(0.1)
    
    frame = getframe(h); 
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    % Write to the GIF File 
    if ii == 1 
        imwrite(imind,cm,fname_gif,'gif', 'Loopcount',inf); 
    else 
        imwrite(imind,cm,fname_gif,'gif','WriteMode','append'); 
    end 
     
end

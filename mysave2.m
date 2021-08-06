function [] = mysave2(fname_save, selfOrgNet, L1v, L2v, L3v, L1H, L2H, L3H, activity_Hint)

L1Pos = [selfOrgNet(1).nx, 0*ones(size(selfOrgNet(1).nx,1),1)];
L2Pos = [selfOrgNet(2).nx, 1*ones(size(selfOrgNet(2).nx,1),1)];
L3Pos = [selfOrgNet(3).nx, 2*ones(size(selfOrgNet(3).nx,1),1)];

L1Hint = activity_Hint{1};
L2Hint = activity_Hint{2};




% L4Pos = [selfOrgNet(4).nx, 3*ones(size(selfOrgNet(4).nx,1),1)];
save(fname_save, 'L1v', 'L2v','L3v','L1H','L2H','L3H','L1Pos','L2Pos','L3Pos', 'L1Hint', 'L2Hint' )

% save(fname_save, 'L2v','L3v','L4v','L2H','L3H','L4H','L1Pos','L2Pos','L3Pos','L4Pos')

end
function [] = mysave_supervised(fname_save, selfOrgNet, L2v, L3v, L2H, L3H, p_k, acc_ctrl, acc_train, acc_test)

L1Pos = [selfOrgNet(1).nx, 0*ones(size(selfOrgNet(1).nx,1),1)];
L2Pos = [selfOrgNet(2).nx, 1*ones(size(selfOrgNet(2).nx,1),1)];
L3Pos = [selfOrgNet(3).nx, 2*ones(size(selfOrgNet(3).nx,1),1)];
% L4Pos = [selfOrgNet(4).nx, 3*ones(size(selfOrgNet(4).nx,1),1)];
save(fname_save, 'L2v','L3v','L2H','L3H','L1Pos','L2Pos','L3Pos', 'p_k','acc_ctrl', 'acc_train', 'acc_test')

end
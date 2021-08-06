function [] = mysave(fname, selfOrgNet, netLossVal, J_vec, lr_vec)

    save(fname, 'selfOrgNet', 'netLossVal','J_vec','lr_vec')
    
end
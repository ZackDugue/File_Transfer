function [y,idx] = maxk(A, k)
[A, idx] = sort(A,'descend');   % No UNIQUE here
y = A(1:k);
idx = idx(1:k);
end
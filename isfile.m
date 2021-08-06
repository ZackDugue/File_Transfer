function [val] = isfile(fname)

val = exist(fname, 'file');

end
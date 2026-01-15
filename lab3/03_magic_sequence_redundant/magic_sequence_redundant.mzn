include "distribute.mzn";

int: N;
set of int: Indexes = 0..N-1;

array[Indexes] of var Indexes: magic;

constraint distribute(magic, array1d(Indexes, Indexes), magic);
% Alternatively:
% constraint forall(i in Indexes)(
%   magic[i] = sum([magic[j] = i | j in Indexes])
% or even
%   magic[i] = count(magic, i)
% );

% Redundant constraint 1: Sum of all elements must equal N
% This is redundant because we're counting N elements total
constraint sum(magic) = N;

% Redundant constraint 2: Weighted sum (index × value) must equal N
% This is redundant because sum of (i × count_of_i) = total_elements = N
constraint sum(i in Indexes)(i * magic[i]) = N;

solve satisfy;
  
output [ "magic sequence = ", show(magic),";\n"];

% Length of the sequence
int: N;

%%%%%%%%%%%%%%
% TODO: VARIABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The sequence itself, call it 'magic'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
array[0..N-1] of var 0..N-1: magic;

%%%%%%%%%%%%%%%%%%%%
% TODO: Constraints
%%%%%%%%%%%%%%%%%%%%

% For each index i, magic[i] equals the number of occurrences of value i in the sequence
constraint forall(i in 0..N-1)(
  magic[i] = sum(j in 0..N-1)(magic[j] = i)
);

% Optimization: the sum of all elements must equal N
constraint sum(i in 0..N-1)(magic[i]) = N;

% Optimization: the sum of i*magic[i] must also equal N
constraint sum(i in 0..N-1)(i * magic[i]) = N;

solve satisfy;
  
%%%%%%%%%%%%%%%%%%%%%%%%
% EXAMPLE OUTPUT  
%%%%%%%%%%%%%%%%%%%%%%%%
% - magic - magic sequence     
%%%%%%%%%%%%%%%%%%%%%%%%
output [ "magic sequence = ", show(magic),";\n"];

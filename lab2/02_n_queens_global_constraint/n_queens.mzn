% number of queens
int: N;

include "alldifferent.mzn";

%%%%%%%%%%%%%
% TODO: VARIABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add variables with correct domains
% tip:
% var 1..N: x; declares a variable with domain 1..N
% use names from the oput
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
array[1..N] of var 1..N: rows;

%%%%%%%%%%%%%
% TODO: CONSTRAINTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add global constraints
% tip: do not use the != constraint, just alldifferent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% All queens in different rows (using alldifferent)
constraint alldifferent(rows);

% No two queens on the same diagonal (positive slope)
% rows[i] + i must be different for all i
constraint alldifferent([rows[i] + i | i in 1..N]);

% No two queens on the same diagonal (negative slope)
% rows[i] - i must be different for all i
constraint alldifferent([rows[i] - i | i in 1..N]);

solve satisfy; 
%%%%%%%%%%%%%%%%%%%
% OUTPUT EXAMPLE  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - rows[i] - queen's position in the ith column 
% - N - number of queens                     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output [ if fix(rows[j]) == i then "|Q" else "|_" endif ++
  if j == N then "\n" else "" endif | i,j in 1..N];

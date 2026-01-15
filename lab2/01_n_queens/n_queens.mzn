% number of queens
int: N;

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
% add constraints
% tip:
% rows[i] = rows[j] (-/+) (j - i)
% matches queens placed on the same diagonal, where:
% - i,j - column index
% - rows[n] - queen's position in the nth col
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% All queens in different rows
constraint forall(i, j in 1..N where i < j)(rows[i] != rows[j]);

% No two queens on the same diagonal (positive slope)
constraint forall(i, j in 1..N where i < j)(rows[i] != rows[j] + (j - i));

% No two queens on the same diagonal (negative slope)
constraint forall(i, j in 1..N where i < j)(rows[i] != rows[j] - (j - i));


solve satisfy; 
  
%%%%%%%%%%%%%%%%%%%
% OUTPUT EXAMPLE  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - rows[i] - queen's position in the ith column 
% - N - number of queens                     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output [ if fix(rows[j]) == i then "|Q" else "|_" endif ++
  if j == N then "\n" else "" endif | i,j in 1..N];

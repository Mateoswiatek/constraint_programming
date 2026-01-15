%%%%%%%%%%%
% REMINDER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% There's something missing here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
include "alldifferent.mzn";

%%%%%%%%%%%
% PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial state of the board
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
array[int,int] of int: board;
set of int: PuzzleRange = index_set_1of2(board);
int: N = max(PuzzleRange);
int: S = floor(sqrt(N));

constraint assert(index_set_2of2(board) = PuzzleRange, "The board should be a square array"); 
constraint assert(S * S == N, "The board size should be a square of a natural number");
constraint assert(min(board) = 0, "The board should contain at least one empty space and no negative values");
constraint assert(max(board) <= N, "The board shouldn't contain values bigger than the board size");

%%%%%%%%%%%
% VARIABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You know... Use the 'puzzle' name from the output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
array[PuzzleRange, PuzzleRange] of var 1..N: puzzle;

%%%%%%%%%%%%%%%%
% Constraints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - you know the sudoku rules
% - remember, that you have to preserve the initial state of the board
% tips.
% - you can slice arrays like in python, e.g. board[1,1..3]
%   is the first three elements of the first row of the board array
% - there are
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Preserve the initial state of the board
constraint forall(i, j in PuzzleRange where board[i,j] > 0)(
  puzzle[i,j] = board[i,j]
);

% Each row must contain all different values
constraint forall(i in PuzzleRange)(
  alldifferent([puzzle[i,j] | j in PuzzleRange])
);

% Each column must contain all different values
constraint forall(j in PuzzleRange)(
  alldifferent([puzzle[i,j] | i in PuzzleRange])
);

% Each SxS sector must contain all different values
constraint forall(a, b in 0..S-1)(
  alldifferent([puzzle[i,j] | i in a*S+1..(a+1)*S, j in b*S+1..(b+1)*S])
);

solve satisfy;

%%%%%%%%%%%%%%%%%%%%
% OUTPUT EXAMPLE  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - puzzle - 2d array of nxn variables
% - N - size of the board, equal S*S         
% - S - size of a single sector            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output [ show(puzzle[i,j]) ++ " " ++
  if j mod S == 0 then " " else "" endif ++ if j == N /\ i != N then
  if i mod S == 0 then "\n\n" else "\n" endif else "" endif
    | i,j in PuzzleRange ] ++ ["\n"];

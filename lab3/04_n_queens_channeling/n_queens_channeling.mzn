include "alldifferent.mzn";
include "lex_lesseq.mzn";

int: N;
set of int: indexes = 1..N;

% First model: rows[i] = column position of queen in row i
array[indexes] of var indexes: rows;

constraint alldifferent(rows);
constraint alldifferent([rows[i] + i | i in indexes]);
constraint alldifferent([rows[i] - i | i in indexes]);

% Second model: qb[i,j] = true if there's a queen at position (i,j)
array[1..N, 1..N] of var bool: qb;

% Constraints for the boolean model - each row has exactly one queen
constraint forall(i in 1..N)(
  sum(j in 1..N)(qb[i,j]) = 1
);

% Each column has exactly one queen
constraint forall(j in 1..N)(
  sum(i in 1..N)(qb[i,j]) = 1
);

% Diagonal constraints: no two queens on same diagonal (top-left to bottom-right)
constraint forall(k in 2-N..N-2)(
  sum(i in 1..N, j in 1..N where i-j = k)(qb[i,j]) <= 1
);

% Diagonal constraints: no two queens on same diagonal (top-right to bottom-left)
constraint forall(k in 2..2*N-1)(
  sum(i in 1..N, j in 1..N where i+j = k)(qb[i,j]) <= 1
);

% Channeling constraints: connect the two models
constraint forall(i in 1..N)(
  forall(j in 1..N)(
    qb[i,j] <-> (rows[i] = j)
  )
);

% Symmetry breaking using lex_lesseq on different permutations of the qb array
% These represent the 8 symmetries of the chessboard:
% qb0: original (identity)
% qb1: reflection over main diagonal (transpose)
% qb2: vertical flip
% qb3: horizontal flip
% qb4: horizontal flip + transpose
% qb5: vertical flip + transpose
% qb6: 180° rotation
% qb7: 180° rotation + transpose

array[int] of var bool: qb0 = array1d(qb);
array[int] of var bool: qb1 = [ qb[j,i] | i,j in 1..N ];
array[int] of var bool: qb2 = [ qb[i,j] | i in reverse(1..N), j in 1..N ];
array[int] of var bool: qb3 = [ qb[j,i] | i in 1..N, j in reverse(1..N) ];
array[int] of var bool: qb4 = [ qb[i,j] | i in 1..N, j in reverse(1..N) ];
array[int] of var bool: qb5 = [ qb[j,i] | i in reverse(1..N), j in 1..N ];
array[int] of var bool: qb6 = [ qb[i,j] | i,j in reverse(1..N) ];
array[int] of var bool: qb7 = [ qb[j,i] | i,j in reverse(1..N) ];

constraint lex_lesseq(qb0, qb1);
constraint lex_lesseq(qb0, qb2);
constraint lex_lesseq(qb0, qb3);
constraint lex_lesseq(qb0, qb4);
constraint lex_lesseq(qb0, qb5);
constraint lex_lesseq(qb0, qb6);
constraint lex_lesseq(qb0, qb7);

solve satisfy; 
  
output [ if fix(rows[j]) == i then "|Q" else "|_" endif ++
  if j == N then "\n" else "" endif | i,j in 1..N];
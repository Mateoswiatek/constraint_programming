include "alldifferent.mzn";

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

solve satisfy;

output [ if fix(rows[j]) == i then "|Q" else "|_" endif ++
  if j == N then "\n" else "" endif | i,j in 1..N];
include "alldifferent.mzn";

int: N;
set of int: indexes = 1..N;

% First model: rows[i] = column position of queen in row i
array[indexes] of var indexes: rows;

constraint alldifferent(rows);
constraint alldifferent([rows[i] + i | i in indexes]);
constraint alldifferent([rows[i] - i | i in indexes]);

solve satisfy;

output [ if fix(rows[j]) == i then "|Q" else "|_" endif ++
  if j == N then "\n" else "" endif | i,j in 1..N];
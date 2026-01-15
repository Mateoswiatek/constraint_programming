include "lex_less.mzn";


% First custom predicate -> `predicate` allows to give names to more complex constraints
predicate knapsack(array[int] of var int: Weights, array[int] of var 0..1: Taken, int: Capacity) =
           sum(i in index_set(Weights))( Weights[i] * Taken[i]) <= Capacity
;

int: items_n;
int: knapsacks_n; 
int: capacity;

array[1..items_n] of int: values;
array[1..items_n] of int: weights;
array[1..knapsacks_n, 1..items_n] of var 0..1: taken;

% Może być wzięty co najwyżej raz
constraint forall(i in 1..items_n)(
 sum(taken[..,i]) <= 1
);

% Użycie naszego customowego predykatu dla każdego "plecaka"
constraint forall(i in 1..knapsacks_n)(
 knapsack(weights, [taken[i,j] | j in 1..items_n], capacity)
);

% Symmetry breaking: impose lexicographic ordering on knapsack rows
% This ensures that knapsack[i] <= knapsack[i+1] lexicographically
% Breaking the symmetry of identical knapsacks
constraint forall(i in 1..knapsacks_n-1)(
 lex_less([taken[i,j] | j in 1..items_n], [taken[i+1,j] | j in 1..items_n])
);

solve maximize sum(i in 1..knapsacks_n, j in 1..items_n) (taken[i,j] * values[j]);

output [if j == 1 then show(i) ++ ") " else "" endif ++ show(taken[i,j]) ++ if j = items_n then "\n" else "" endif  | i in 1..knapsacks_n, j in 1..items_n] 
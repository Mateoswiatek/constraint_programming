% parameters
int: nodesNumber;
array [1..nodesNumber, 1..nodesNumber] of bool: edges;

% optional parameters for symmetry breaking
int: minColorsNumber;
array [int] of int: maxClique;
 
% decision variables
var 1..nodesNumber: colorsNumber;
array [1..nodesNumber] of var 1..nodesNumber: nodes;

% constraints
constraint colorsNumber = max(nodes);

constraint forall (i,j in 1..nodesNumber where i < j /\ edges[i,j])(
  nodes[i] != nodes[j]
);

% symmetry breaking using clique information
% The clique provides a lower bound on colors needed
constraint if minColorsNumber > 0 then colorsNumber >= minColorsNumber else true endif;

constraint if length(maxClique) > 0 then
  % Assign consecutive colors to clique nodes: color 1, 2, 3, ..., k
  % This breaks symmetry by fixing the color assignment for the largest clique
  forall(i in 1..length(maxClique))(
    nodes[maxClique[i]] = i
  )
else
  true
endif;

% goal
solve minimize colorsNumber;

% output
output [show(colorsNumber), " colors\n",] ++
       [show(nodes[i]) ++ " " | i in 1..nodesNumber]
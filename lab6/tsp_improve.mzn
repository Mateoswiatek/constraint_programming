include "globals.mzn";

int: Nodes;
set of int: NUM = 1..Nodes;
array[NUM,NUM] of int: Dist;
array[NUM] of var NUM: route;
array[NUM] of 0..Nodes: initial_route;

var int: total_distance;

% Cities in the route should not repeat (each city visited exactly once)
constraint alldifferent(route);

% Fix the first city to 1 (break symmetry)
constraint route[1] = 1;

% Fix variables from initial_route that are not zero (like sudoku)
constraint forall(i in NUM where initial_route[i] != 0)(
    route[i] = initial_route[i]
);

% Calculate the total distance based on the route
constraint total_distance = sum(i in 1..Nodes-1)(Dist[route[i], route[i+1]]) + Dist[route[Nodes], route[1]];

% Minimize total distance
solve minimize total_distance;

output ["total distance: \(total_distance)\nroute = \(route)"];
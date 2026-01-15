include "globals.mzn";

int: Nodes;
set of int: NUM = 1..Nodes;
array[NUM,NUM] of int: Dist;
array[NUM] of var NUM: route;

var int: total_distance;

% Cities in the route should not repeat (each city visited exactly once)
constraint alldifferent(route);

% Fix the first city to 1 (break symmetry)
constraint route[1] = 1;

% Calculate the total distance based on the route
constraint total_distance = sum(i in 1..Nodes-1)(Dist[route[i], route[i+1]]) + Dist[route[Nodes], route[1]];

% Just satisfy the constraints (find any valid solution)
solve satisfy;

output ["total distance: \(total_distance)\nroute = \(route)"];
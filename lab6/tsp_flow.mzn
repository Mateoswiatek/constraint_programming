include "globals.mzn";

int: Nodes;
set of int: NUM = 1..Nodes;
array[NUM,NUM] of int: Dist;

array[NUM,NUM] of var bool: flow;
array[NUM,NUM] of var 0..Nodes-1: visit_counter;

var int: total_distance;

% a) Flow constraints:
% 1) Total flow coming from every node should be equal 1
constraint forall(i in NUM)(sum(j in NUM)(flow[i,j]) = 1);

% 2) Total flow going into every node should be equal 1
constraint forall(j in NUM)(sum(i in NUM)(flow[i,j]) = 1);

% No self-loops
constraint forall(i in NUM)(flow[i,i] = false);

% b) Visit counter constraints:
% 1) Visit counter for edges originating from the first node is 0
constraint forall(j in NUM)(visit_counter[1,j] = 0);

% 2) Visit counter for inactive edges should be equal 0
% Use multiplication to make it linear: visit_counter <= flow * (Nodes-1)
constraint forall(i,j in NUM where i != 1)(
    visit_counter[i,j] <= bool2int(flow[i,j]) * (Nodes - 1)
);

% 3) Total visit counter for edges ending at the first node should equal Nodes - 1
constraint sum(i in NUM)(visit_counter[i,1]) = Nodes - 1;

% 4) For every normal node (not the first node), the sum of visit counters for outgoing edges
%    has to be bigger than sum of visit counters for incoming edges exactly by 1
constraint forall(k in 2..Nodes)(
    sum(j in NUM)(visit_counter[k,j]) = sum(i in NUM)(visit_counter[i,k]) + 1
);

% c) Objective: calculate total distance based on flow
constraint total_distance = sum(i,j in NUM)(bool2int(flow[i,j]) * Dist[i,j]);

% Minimize total distance
solve minimize total_distance;

output ["total distance: \(total_distance)\n"] ++
       [if fix(flow[i,j]) then " -> \(j)" else "" endif | i, j in NUM];
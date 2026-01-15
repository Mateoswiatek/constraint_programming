include "diffn.mzn";
include "cumulative.mzn";

% Parameters
%%%%%%%%%%%
int: n;                      % How many squares do we have?
set of int: SQUARES = 1..n;  % Set of the available squares

% Variables
%%%%%%%%%%%
% Minimum container dimension is n (largest square), maximum is sum of all squares in one dimension
var n..sum(SQUARES): height;    % height of the container
var n..sum(SQUARES): width;     % width of the conainer
var int: area = height * width; % container's area
array[SQUARES] of var 0..sum(SQUARES)-n: x; % squares' coordinates in the container (0-indexed)
array[SQUARES] of var 0..sum(SQUARES)-n: y; % squares' coordinated in the container (0-indexed)
  
% Constraints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Constraint 1: Squares should fit inside the container
constraint forall(i in SQUARES) (
  x[i] + i <= width /\ y[i] + i <= height
);

% Constraint 2: Squares should not overlap (using diffn global constraint)
constraint diffn(x, y, [i | i in SQUARES], [i | i in SQUARES]);

% Redundant Constraints: Using cumulative to model packing as 1D scheduling
% Horizontal dimension: treat x-coordinates as start times, widths as durations
constraint cumulative(x, [i | i in SQUARES], [i | i in SQUARES], height);
% Vertical dimension: treat y-coordinates as start times, heights as durations
constraint cumulative(y, [i | i in SQUARES], [i | i in SQUARES], width);

% Goal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stage 5: Search annotations
% 1. Try smallest containers first (minimize height and width)
% 2. Place biggest squares first (reverse order: n, n-1, ..., 1)
solve :: seq_search([
    int_search([height, width], input_order, indomain_min, complete),
    int_search([x[i] | i in reverse(SQUARES)] ++ [y[i] | i in reverse(SQUARES)],
               input_order, indomain_min, complete)
  ])
  minimize area; 
  

% Boring output  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output [ show(i) ++ " > (" ++ show(x[i]) ++ "," ++ show(y[i]) ++ ")\n" | i in 1..n] ++
  ["area = " ++ show(width) ++ " * " ++ show(height) ++ " = " ++ show(area)];
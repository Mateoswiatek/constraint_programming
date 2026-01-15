include "globals.mzn";

int: nr;
set of int: RECLAIMER = 1..nr;
bool: stageF;

int: ns;
set of int: STOCKPILE = 1..ns;
array[STOCKPILE] of int: size;
array[STOCKPILE] of SHIP: ship;

int: maxtime;
set of int: TIME = 0..maxtime;

int: nsh;
set of int: SHIP = 1..nsh;
array[SHIP] of TIME: arrival;

int: len;
set of int: POSITION = 0..len;

int: stack_time;
int: reclaim_time;
int: reclaim_speed;

array[STOCKPILE] of var POSITION: westend;
array[STOCKPILE] of var POSITION: eastend;
array[STOCKPILE] of var TIME: stack;
array[STOCKPILE] of var TIME: endstack;
array[STOCKPILE] of var TIME: reclaim;
array[STOCKPILE] of var TIME: finished;
array[STOCKPILE] of var RECLAIMER: which;

array[STOCKPILE] of int: stack_dur = [size[s] * stack_time | s in STOCKPILE];
array[STOCKPILE] of int: reclaim_dur = [size[s] * reclaim_time | s in STOCKPILE];

% ============================================================================
% Stage A - Packing
% ============================================================================

constraint forall(s in STOCKPILE)(eastend[s] = westend[s] + size[s]);
constraint forall(s in STOCKPILE)(eastend[s] <= len);
constraint forall(s in STOCKPILE)(endstack[s] = stack[s] + stack_dur[s]);
constraint forall(s in STOCKPILE)(endstack[s] <= reclaim[s]);
constraint forall(s in STOCKPILE)(finished[s] = reclaim[s] + reclaim_dur[s]);
constraint forall(s in STOCKPILE)(finished[s] <= maxtime);

constraint diffn(westend, stack, size, [finished[s] - stack[s] | s in STOCKPILE]);

% ============================================================================
% Stage B/D - Reclaimer Assignment + Movement Time
% ============================================================================

constraint forall(s1, s2 in STOCKPILE where s1 < s2)(
  which[s1] = which[s2] ->
  let { var int: travel = abs(westend[s1] - westend[s2]) * reclaim_speed } in
  (finished[s1] + travel <= reclaim[s2] \/ finished[s2] + travel <= reclaim[s1])
);

% ============================================================================
% Stage C - Ship Constraints
% ============================================================================

constraint forall(s in STOCKPILE)(reclaim[s] >= arrival[ship[s]]);

constraint forall(s1, s2 in STOCKPILE where s1 < s2)(
  ship[s1] = ship[s2] -> (finished[s1] <= reclaim[s2] \/ finished[s2] <= reclaim[s1])
);

% ============================================================================
% Stage E - Objective
% ============================================================================

array[SHIP] of var TIME: ship_finish;
constraint forall(sh in SHIP)(
  ship_finish[sh] = max([if ship[s] = sh then finished[s] else 0 endif | s in STOCKPILE])
);

var int: obj = sum(sh in SHIP)(ship_finish[sh] - arrival[sh]);

% ============================================================================
% Stage F - Rail Constraints
% ============================================================================

constraint stageF -> forall(i in 1..nr div 2, s1 in STOCKPILE, s2 in STOCKPILE)(
  (which[s1] = 2*i - 1 /\ which[s2] = 2*i /\
   reclaim[s1] < finished[s2] /\ reclaim[s2] < finished[s1]) ->
  westend[s1] <= westend[s2]
);

constraint stageF -> forall(i in 1..nr div 2)(
  forall(s1 in STOCKPILE, s3 in STOCKPILE where s1 != s3)(
    forall(s2 in STOCKPILE)(
      (which[s1] = 2*i - 1 /\ which[s3] = 2*i - 1 /\ which[s2] = 2*i /\
       finished[s1] <= reclaim[s3] /\
       reclaim[s2] <= finished[s1] /\ reclaim[s3] <= finished[s2]) ->
      max(westend[s1], westend[s3]) <= westend[s2]
    )
  )
);

constraint stageF -> forall(i in 1..nr div 2)(
  forall(s2 in STOCKPILE, s4 in STOCKPILE where s2 != s4)(
    forall(s1 in STOCKPILE)(
      (which[s2] = 2*i /\ which[s4] = 2*i /\ which[s1] = 2*i - 1 /\
       finished[s2] <= reclaim[s4] /\
       reclaim[s1] <= finished[s2] /\ reclaim[s4] <= finished[s1]) ->
      westend[s1] <= min(westend[s2], westend[s4])
    )
  )
);

% ============================================================================
% Reclaimer Positions (required by checker)
% ============================================================================

array[RECLAIMER, TIME] of var POSITION: reclaimers_positions;

% During reclaiming, position must equal westend of stockpile
% Use explicit reclaimer index to avoid variable indexing issues
constraint forall(r in RECLAIMER, s in STOCKPILE, t in TIME)(
  (which[s] = r /\ reclaim[s] <= t /\ t <= finished[s]) ->
  reclaimers_positions[r, t] = westend[s]
);

% Helper: is reclaimer r active at time t?
array[RECLAIMER, TIME] of var bool: is_reclaiming_at;
constraint forall(r in RECLAIMER, t in TIME)(
  is_reclaiming_at[r,t] = exists(s in STOCKPILE)(which[s] = r /\ reclaim[s] <= t /\ t < finished[s])
);

% Movement speed limit when not reclaiming
constraint forall(r in RECLAIMER, t in 1..maxtime)(
  (not is_reclaiming_at[r,t] /\ not is_reclaiming_at[r,t-1]) ->
  abs(reclaimers_positions[r, t] - reclaimers_positions[r, t-1]) <= reclaim_speed
);

% Initial position when not reclaiming
constraint forall(r in RECLAIMER)(
  not is_reclaiming_at[r,0] -> reclaimers_positions[r, 0] = 0
);

% Stage F: rail ordering at all times
constraint stageF -> forall(i in 1..nr div 2, t in TIME)(
  reclaimers_positions[2*i - 1, t] <= reclaimers_positions[2*i, t]
);

% ============================================================================
% Search Strategy
% ============================================================================

solve
  :: seq_search([
    int_search(which, input_order, indomain_min),
    int_search(reclaim, smallest, indomain_min),
    int_search(westend, first_fail, indomain_min),
    int_search(stack, smallest, indomain_min)
  ])
  minimize obj;

output
["westend  = ", show(westend), ";\n"] ++
["eastend  = ", show(eastend), ";\n"] ++
["stack    = ", show(stack), ";\n"] ++
["endstack = ", show(endstack), ";\n"] ++
["reclaim  = ", show(reclaim), ";\n"] ++
["finish   = ", show(finished) , ";\n"] ++
["which    = ", show(which), ";\n"] ++
["obj = ",show(obj), ";\n"];
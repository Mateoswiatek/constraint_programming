include "cumulative.mzn";

% Parameters
%%%%%%%%%%%%%
set of int: Hour = index_set(hourly_capacities);
set of int: Minute = 0..max(Hour) * 60 - 1;               % -1 because we start counting time from 0
set of int: Student = index_set(visits_preferred_hours);  

array[int] of int: hourly_capacities;                     % how many students can be processed in parallel
                                                          % in every hour

array[int] of Hour: visits_preferred_hours;               % visits_preferred_hours[2] = 3 means that 
                                                          % the student 2 would like to start grading
                                                          % in the the third hour (minutes 120 - 179)

array[Student] of int: visits_durations_min;              % visits_durations_min[3] = 37 means that 
                                                          % grading the student 3 will take 37 minutes

constraint assert(min(visits_durations_min) > 0, "visits should have a positive duration");
constraint assert(min(hourly_capacities) >= 0, "the capacity cannot be negative");
constraint assert(max(hourly_capacities) > 0, "the total capacity has to be positive");


%% CONSTRAINTS:
%  `schedule` represent our result: at what minute each students should be graded
array[Student] of var Minute: schedule;

% Constraint 1: Real students must finish before end of day
constraint forall(s in Student) (
  schedule[s] + visits_durations_min[s] <= max(Minute) + 1
);

% Constraint 2: Create dummy tasks to model varying hourly capacity
% For each hour, create dummy tasks to block (max_capacity - hourly_capacity) slots
int: max_capacity = max(hourly_capacities);

% Build arrays of dummy tasks (one dummy per blocked slot per hour)
array[int] of int: dummy_starts =
  [(h-1) * 60 | h in Hour, slot in 1..(max_capacity - hourly_capacities[h])];
array[int] of int: dummy_durations =
  [60 | h in Hour, slot in 1..(max_capacity - hourly_capacities[h])];

% Constraint 3: Cumulative - combine real students with dummy blockers
constraint cumulative(
  schedule ++ dummy_starts,
  visits_durations_min ++ dummy_durations,
  [1 | s in Student] ++ [1 | d in index_set(dummy_starts)],
  max_capacity
);

%% OBJECTIVE:
% Calculate preferred time window in minutes for each student
% Hour h means minutes [60*(h-1), 60*h - 1]
array[Student] of int: preferred_window_start =
  [60 * (visits_preferred_hours[s] - 1) | s in Student];
array[Student] of int: preferred_window_end =
  [60 * visits_preferred_hours[s] - 1 | s in Student];

% Calculate how early or late each student is scheduled
% too_early: if scheduled before the window start
% too_late: if scheduled after the window end
array[Student] of var Minute: minutes_too_early =
  [max(0, preferred_window_start[s] - schedule[s]) | s in Student];
array[Student] of var Minute: minutes_too_late =
  [max(0, schedule[s] - preferred_window_end[s]) | s in Student];

% Objective: sum of squared violations
var int: objective =
  sum(s in Student)(
    minutes_too_early[s] * minutes_too_early[s] +
    minutes_too_late[s] * minutes_too_late[s]
  );


% Search annotations: prioritize scheduling students close to their preferences
% First ensure students are not too early, then not too late
solve :: seq_search([
    int_search(minutes_too_early, input_order, indomain_min, complete),
    int_search(minutes_too_late, input_order, indomain_min, complete),
    int_search(schedule, first_fail, indomain_min, complete)
  ])
  minimize objective;

output [join("\n", [
  "schedule = \(schedule);",
  "minutes_too_early = \(minutes_too_early);",
  "minutes_too_late = \(minutes_too_late);",
  "objective = \(objective);"
  ])
];


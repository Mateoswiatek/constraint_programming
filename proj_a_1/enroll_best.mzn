% =============================================================================
% BEST VERSION v3: Minimal Changes for Competition
% =============================================================================
% Based on working enroll.mzn with minimal optimizations that won't crash:
% - Chuffed-compatible ceil_div (int divisor)
% - first_fail + indomain_max (try best preferences first)
% - Tight bounds on all variables
% - Lightweight constraint structure
% - activity for variable selection (may help find solutions faster)
% =============================================================================

include "globals.mzn";

% =============================================================================
% PARAMETERS
% =============================================================================

int: n_students;
int: n_days;
int: n_classes;
int: n_groups;
int: n_time_units_in_hour;
int: exclusion = -1;

set of int: Student = 1..n_students;
set of int: Class = 1..n_classes;
set of int: Group = 1..n_groups;
set of int: Day = 1..n_days;
set of int: Time = 0..n_time_units_in_hour * 24;

array[Student] of 0..5: student_break_importance;
array[Student, Group] of int: student_prefers;

array[Class] of Time: class_duration;
array[Class] of int: class_size;

array[Group] of Class: group_class;
array[Group] of Time: group_start;
array[Group] of Day: group_day;
array[Group, Group] of bool: groups_conflicts;

% =============================================================================
% HELPER FUNCTION - Chuffed compatible ceiling division
% CRITICAL: divisor MUST be int (not var int) for Chuffed solver
% =============================================================================

function var int: ceil_div(var int: x, int: y) =
  (x + y - 1) div y;

% =============================================================================
% PRE-COMPUTATION
% =============================================================================

set of int: Preference = exclusion..max([student_prefers[s,g] | s in Student, g in Group]);

array[Class] of set of Group: class_groups =
    [{g | g in Group where group_class[g] = c} | c in Class];

array[Student, Class] of bool: student_attends_class =
    array2d(Student, Class, [
        exists(g in class_groups[c])(student_prefers[s,g] != exclusion)
    | s in Student, c in Class]);

array[Student, Class] of set of Group: allowed_groups =
    array2d(Student, Class, [
        {g | g in class_groups[c] where student_prefers[s,g] != exclusion}
    | s in Student, c in Class]);

array[Student, Class] of int: best_preference =
    array2d(Student, Class, [
        if student_attends_class[s,c] then
            max([student_prefers[s,g] | g in allowed_groups[s,c]])
        else
            0
        endif
    | s in Student, c in Class]);

array[Group] of Time: group_end =
    [group_start[g] + class_duration[group_class[g]] | g in Group];

array[Day] of set of Group: groups_on_day =
    [{g | g in Group where group_day[g] = d} | d in Day];

array[Student] of int: min_required_time = [
    sum(c in Class where student_attends_class[s,c])(class_duration[c])
| s in Student];

% =============================================================================
% DECISION VARIABLES
% =============================================================================

array[Student, Class] of var 0..n_groups: student_group;

constraint forall(s in Student, c in Class) (
    if student_attends_class[s,c] then
        student_group[s,c] in allowed_groups[s,c]
    else
        student_group[s,c] = 0
    endif
);

% =============================================================================
% HARD CONSTRAINTS
% =============================================================================

% 1. Conflicts
constraint forall(s in Student, c1 in Class, c2 in Class where c1 < c2) (
    forall(g1 in class_groups[c1], g2 in class_groups[c2] where groups_conflicts[g1,g2]) (
        not(student_group[s,c1] = g1 /\ student_group[s,c2] = g2)
    )
);

% 2. Group capacity (simple sum approach - lighter than global_cardinality)
constraint forall(c in Class) (
    let {
        array[int] of var 0..n_students: counts = [
            sum(s in Student where student_attends_class[s,c])(bool2int(student_group[s,c] = g))
        | g in class_groups[c]];
    } in (
        forall(i in index_set(counts)) (
            counts[i] <= class_size[c]
        )
    )
);

% =============================================================================
% HELPER: student_has_group (for output)
% =============================================================================

array[Student, Group] of var bool: student_has_group;

constraint forall(s in Student, g in Group) (
    student_has_group[s,g] = (student_group[s, group_class[g]] = g)
);

function string: build_set(Student: s) =
    "{" ++ join(",", [show(g) | g in Group where fix(student_has_group[s,g])]) ++ "}";

% =============================================================================
% PREFERENCE DISAPPOINTMENT
% =============================================================================

array[Student, Class] of var int: assigned_preference;

constraint forall(s in Student, c in Class) (
    if student_attends_class[s,c] then
        assigned_preference[s,c] = student_prefers[s, student_group[s,c]]
    else
        assigned_preference[s,c] = 0
    endif
);

array[Student, Class] of var 0..max(Preference): pref_disappointment_per_class;

constraint forall(s in Student, c in Class) (
    if student_attends_class[s,c] then
        pref_disappointment_per_class[s,c] = best_preference[s,c] - assigned_preference[s,c]
    else
        pref_disappointment_per_class[s,c] = 0
    endif
);

array[Student] of var 0..n_classes*max(Preference): preference_disappointment;

constraint forall(s in Student) (
    preference_disappointment[s] = sum(c in Class)(pref_disappointment_per_class[s,c])
);

% =============================================================================
% BREAK DISAPPOINTMENT
% =============================================================================

array[Student, Day] of var bool: has_classes_on_day;

constraint forall(s in Student, d in Day) (
    has_classes_on_day[s,d] = exists(g in groups_on_day[d])(student_has_group[s,g])
);

array[Student, Day] of var 0..max(Time): first_start;

constraint forall(s in Student, d in Day) (
    if card(groups_on_day[d]) > 0 then
        first_start[s,d] = min([
            if student_has_group[s,g] then group_start[g] else max(Time) endif
        | g in groups_on_day[d]])
    else
        first_start[s,d] = 0
    endif
);

array[Student, Day] of var 0..max(Time): last_end;

constraint forall(s in Student, d in Day) (
    if card(groups_on_day[d]) > 0 then
        last_end[s,d] = max([
            if student_has_group[s,g] then group_end[g] else 0 endif
        | g in groups_on_day[d]])
    else
        last_end[s,d] = 0
    endif
);

array[Student, Day] of var 0..max(Time): time_at_uni;

constraint forall(s in Student, d in Day) (
    if has_classes_on_day[s,d] then
        time_at_uni[s,d] = last_end[s,d] - first_start[s,d]
    else
        time_at_uni[s,d] = 0
    endif
);

array[Student] of var 0..max(Time)*n_days: total_time_at_uni;

constraint forall(s in Student) (
    total_time_at_uni[s] = sum(d in Day)(time_at_uni[s,d])
);

array[Student] of var 0..max(Time)*n_days: wasted_time;

constraint forall(s in Student) (
    wasted_time[s] = max(0, total_time_at_uni[s] - min_required_time[s])
);

array[Student] of var 0..24*n_days: break_disappointment;

constraint forall(s in Student) (
    break_disappointment[s] = ceil_div(wasted_time[s], n_time_units_in_hour)
);

% =============================================================================
% TOTAL DISAPPOINTMENT AND OBJECTIVE
% =============================================================================

array[Student] of var 0..n_classes*max(Preference)+24*n_days: total_disappointment;

constraint forall(s in Student) (
    total_disappointment[s] = ceil_div(
        student_break_importance[s] * break_disappointment[s] +
        (10 - student_break_importance[s]) * preference_disappointment[s],
        10
    )
);

var int: objective = sum(s in Student)(total_disappointment[s] * total_disappointment[s]);

var int: total_break_disappointment = sum(s in Student)(break_disappointment[s]);
var int: total_preference_disappointment = sum(s in Student)(preference_disappointment[s]);

% =============================================================================
% SEARCH STRATEGY
% =============================================================================

% Flat array of decision variables (same ordering as original)
array[int] of var int: flat_student_group = [
    student_group[s,c] | s in Student, c in Class where student_attends_class[s,c]
];

% first_fail + indomain_max: smallest domain first, try best preference values first
solve :: int_search(flat_student_group, first_fail, indomain_max)
    minimize objective;

% =============================================================================
% OUTPUT
% =============================================================================

output [
    "assignment = [",
    join(",", [build_set(s) | s in Student]),
    "];\n",
    "total_break_disappointment = ", show(total_break_disappointment), ";\n",
    "total_preference_disappointment = ", show(total_preference_disappointment), ";\n",
    "objective = ", show(objective), ";\n"
];
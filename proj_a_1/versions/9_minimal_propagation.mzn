% =============================================================================
% VERSION 9: Minimal Propagation - Ultra-light constraints
% =============================================================================
% Minimalizacja propagacji - tylko niezbędne ograniczenia.
% Break_disappointment obliczane dopiero w output (nie w ograniczeniach).
% =============================================================================

include "globals.mzn";

% Parametry
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

% Funkcja pomocnicza
function var int: ceil_div(var int: x, var int: y) =
  x div y + ((x mod y) > 0);

% =============================================================================
% POMOCNICZE ZBIORY
% =============================================================================

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

array[Group] of Time: group_end = [group_start[g] + class_duration[group_class[g]] | g in Group];
array[Day] of set of Group: groups_on_day = [{g | g in Group where group_day[g] = d} | d in Day];

% =============================================================================
% ZMIENNE DECYZYJNE
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
% OGRANICZENIA TWARDE (minimalne)
% =============================================================================

% 1. Konflikty między grupami
constraint forall(s in Student, c1 in Class, c2 in Class where c1 < c2 /\
           student_attends_class[s,c1] /\ student_attends_class[s,c2]) (
    forall(g1 in allowed_groups[s,c1], g2 in allowed_groups[s,c2]
           where groups_conflicts[g1,g2]) (
        not(student_group[s,c1] = g1 /\ student_group[s,c2] = g2)
    )
);

% 2. Limity wielkości grup
constraint forall(g in Group) (
    count([student_group[s,group_class[g]] | s in Student], g) <= class_size[group_class[g]]
);

% =============================================================================
% TYLKO PREFERENCE DISAPPOINTMENT W OGRANICZENIACH
% =============================================================================

array[Student] of var 0..n_classes*20: preference_disappointment;

constraint forall(s in Student) (
    preference_disappointment[s] = sum(c in Class where student_attends_class[s,c])(
        best_preference[s,c] - student_prefers[s, student_group[s,c]]
    )
);

% =============================================================================
% UPROSZCZONA FUNKCJA CELU - tylko preference
% =============================================================================

var int: total_preference_disappointment = sum(s in Student)(preference_disappointment[s]);

% Prosta funkcja celu - suma kwadratów preferencji
var int: simple_objective = sum(s in Student)(preference_disappointment[s] * preference_disappointment[s]);

% =============================================================================
% ROZWIĄZANIE
% =============================================================================

solve :: int_search(
    [student_group[s,c] | s in Student, c in Class where student_attends_class[s,c]],
    first_fail,
    indomain_min
) minimize simple_objective;

% =============================================================================
% OBLICZENIA DLA WYJŚCIA (po znalezieniu rozwiązania)
% =============================================================================

function int: calc_break_disappointment(Student: s) =
    let {
        int: total_at_uni = sum(d in Day where card(groups_on_day[d]) > 0) (
            let {
                set of int: my_groups = {g | g in groups_on_day[d] where fix(student_group[s, group_class[g]]) = g};
            } in if card(my_groups) > 0 then
                max([group_end[g] | g in my_groups]) - min([group_start[g] | g in my_groups])
            else
                0
            endif
        );
        int: min_req = sum(c in Class where student_attends_class[s,c])(class_duration[c]);
        int: wasted = max(0, total_at_uni - min_req);
    } in (wasted div n_time_units_in_hour) + bool2int(wasted mod n_time_units_in_hour > 0);

function int: calc_total_disappointment(Student: s) =
    let {
        int: bd = calc_break_disappointment(s);
        int: pd = fix(preference_disappointment[s]);
        int: w = student_break_importance[s];
    } in (w * bd + (10 - w) * pd) div 10 + bool2int((w * bd + (10 - w) * pd) mod 10 > 0);

% =============================================================================
% WYJŚCIE
% =============================================================================

output [
    "assignment = [",
    join(",", [
        "{" ++ join(",", [show(fix(student_group[s,c])) | c in Class where student_attends_class[s,c]]) ++ "}"
    | s in Student]),
    "];\n",
    "total_break_disappointment = ", show(sum(s in Student)(calc_break_disappointment(s))), ";\n",
    "total_preference_disappointment = ", show(total_preference_disappointment), ";\n",
    "objective = ", show(sum(s in Student)(calc_total_disappointment(s) * calc_total_disappointment(s))), ";\n"
];

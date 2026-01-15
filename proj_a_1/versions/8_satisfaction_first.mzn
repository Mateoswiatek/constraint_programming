% =============================================================================
% VERSION 8: Satisfaction First - Find ANY valid solution
% =============================================================================
% Ultra-prosta wersja - tylko znajdź JAKIEKOLWIEK rozwiązanie.
% Bez optymalizacji, minimalna propagacja, proste wyjście.
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
% OGRANICZENIA TWARDE
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
    sum(s in Student)(bool2int(student_group[s,group_class[g]] = g)) <= class_size[group_class[g]]
);

% =============================================================================
% OBLICZENIA DLA WYJŚCIA
% =============================================================================

array[Student] of var int: preference_disappointment;

constraint forall(s in Student) (
    preference_disappointment[s] = sum(c in Class where student_attends_class[s,c])(
        best_preference[s,c] - student_prefers[s, student_group[s,c]]
    )
);

array[Student] of var int: break_disappointment;

constraint forall(s in Student) (
    let {
        var int: total_at_uni = sum(d in Day where card(groups_on_day[d]) > 0) (
            let {
                array[int] of var int: starts = [
                    if student_group[s, group_class[g]] = g then group_start[g] else max(Time) endif
                | g in groups_on_day[d]];
                array[int] of var int: ends = [
                    if student_group[s, group_class[g]] = g then group_end[g] else 0 endif
                | g in groups_on_day[d]];
                var bool: has_any = exists(g in groups_on_day[d])(student_group[s, group_class[g]] = g);
            } in if has_any then max(ends) - min(starts) else 0 endif
        );
        int: min_req = sum(c in Class where student_attends_class[s,c])(class_duration[c]);
        var int: wasted = max(0, total_at_uni - min_req);
    } in (
        break_disappointment[s] = ceil_div(wasted, n_time_units_in_hour)
    )
);

array[Student] of var int: total_disappointment;

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
% ROZWIĄZANIE - TYLKO SATISFY
% =============================================================================

solve satisfy;

% =============================================================================
% WYJŚCIE
% =============================================================================

output [
    "assignment = [",
    join(",", [
        "{" ++ join(",", [show(student_group[s,c]) | c in Class where student_attends_class[s,c]]) ++ "}"
    | s in Student]),
    "];\n",
    "total_break_disappointment = ", show(total_break_disappointment), ";\n",
    "total_preference_disappointment = ", show(total_preference_disappointment), ";\n",
    "objective = ", show(objective), ";\n"
];

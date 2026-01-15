% =============================================================================
% VERSION 5: Advanced Search Strategies
% =============================================================================
% Dodajemy:
% - dom_w_deg jako strategię wyboru zmiennych (lepsza niż first_fail)
% - Redundantne ograniczenia dla lepszej propagacji
% - Uporządkowanie zmiennych wg "trudności" (studenci z mniejszą liczbą opcji najpierw)
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
set of int: WorkTime = 0..max(Time)*n_days;
set of int: Preference = exclusion..max([student_prefers[s,g] | s in Student, g in Group]);

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
% POMOCNICZE ZBIORY I TABLICE
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

% Liczba dozwolonych grup dla każdego studenta (miara "trudności")
array[Student] of int: student_flexibility = [
    sum(c in Class)(card(allowed_groups[s,c]))
| s in Student];

% Sortowanie studentów po trudności (mniej elastyczni najpierw)
% To jest heurystyka - przydzielamy najpierw tych z mniejszym wyborem
array[int] of Student: students_by_difficulty =
    sort_by(Student, [student_flexibility[s] | s in Student]);

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
constraint forall(s in Student, c1 in Class, c2 in Class where c1 < c2) (
    forall(g1 in class_groups[c1], g2 in class_groups[c2] where groups_conflicts[g1,g2]) (
        not(student_group[s,c1] = g1 /\ student_group[s,c2] = g2)
    )
);

% 2. Limity wielkości grup
array[Group] of var 0..n_students: group_count;

constraint forall(g in Group) (
    group_count[g] = sum(s in Student, c in Class where g in class_groups[c])(
        bool2int(student_group[s,c] = g)
    )
);

constraint forall(g in Group) (
    group_count[g] <= class_size[group_class[g]]
);

% =============================================================================
% REDUNDANTNE OGRANICZENIA (dla lepszej propagacji)
% =============================================================================

% Suma studentów w grupach klasy = liczba studentów uczęszczających na klasę
constraint redundant_constraint(
    forall(c in Class) (
        sum(g in class_groups[c])(group_count[g]) =
        sum(s in Student)(bool2int(student_attends_class[s,c]))
    )
);

% =============================================================================
% ZBIORY DLA WYJŚCIA
% =============================================================================

array[Student] of var set of Group: assignment;

constraint forall(s in Student) (
    assignment[s] = {g | c in Class, g in class_groups[c] where student_group[s,c] = g}
);

% =============================================================================
% OBLICZENIE PREFERENCE DISAPPOINTMENT
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

array[Student] of var int: preference_disappointment;

constraint forall(s in Student) (
    preference_disappointment[s] = sum(c in Class)(pref_disappointment_per_class[s,c])
);

% =============================================================================
% OBLICZENIE BREAK DISAPPOINTMENT
% =============================================================================

array[Student, Day] of var Time: first_start;
array[Student, Day] of var Time: last_end;
array[Student, Day] of var bool: has_classes_on_day;

constraint forall(s in Student, d in Day) (
    has_classes_on_day[s,d] = exists(c in Class where student_attends_class[s,c])(
        student_group[s,c] != 0 /\ group_day[student_group[s,c]] = d
    )
);

constraint forall(s in Student, d in Day) (
    if has_classes_on_day[s,d] then
        first_start[s,d] = min(c in Class where student_attends_class[s,c] /\ group_day[student_group[s,c]] = d)(
            group_start[student_group[s,c]]
        )
    else
        first_start[s,d] = 0
    endif
);

constraint forall(s in Student, d in Day) (
    if has_classes_on_day[s,d] then
        last_end[s,d] = max(c in Class where student_attends_class[s,c] /\ group_day[student_group[s,c]] = d)(
            group_end[student_group[s,c]]
        )
    else
        last_end[s,d] = 0
    endif
);

array[Student, Day] of var int: time_at_uni;

constraint forall(s in Student, d in Day) (
    if has_classes_on_day[s,d] then
        time_at_uni[s,d] = last_end[s,d] - first_start[s,d]
    else
        time_at_uni[s,d] = 0
    endif
);

array[Student] of var int: total_time_at_uni;

constraint forall(s in Student) (
    total_time_at_uni[s] = sum(d in Day)(time_at_uni[s,d])
);

array[Student] of int: min_required_time = [
    sum(c in Class where student_attends_class[s,c])(class_duration[c])
| s in Student];

array[Student] of var int: wasted_time;

constraint forall(s in Student) (
    wasted_time[s] = max(0, total_time_at_uni[s] - min_required_time[s])
);

array[Student] of var int: break_disappointment;

constraint forall(s in Student) (
    break_disappointment[s] = ceil_div(wasted_time[s], n_time_units_in_hour)
);

% =============================================================================
% OBLICZENIE FUNKCJI CELU
% =============================================================================

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
% STRATEGIA PRZESZUKIWANIA
% =============================================================================

% Uporządkuj zmienne wg trudności studenta, potem wg klasy
array[int] of var int: ordered_vars = [
    student_group[s,c]
    | s in students_by_difficulty, c in Class
    where student_attends_class[s,c]
];

% dom_w_deg: domain size / weighted degree - bardzo efektywna heurystyka
% indomain_min: ale mogłoby być indomain z preferencjami
solve :: int_search(
    ordered_vars,
    dom_w_deg,
    indomain_min
) minimize objective;

% =============================================================================
% WYJŚCIE
% =============================================================================

output [
    "assignment = [",
    join(",", [show(assignment[s]) | s in Student]),
    "];\n",
    "total_break_disappointment = ", show(total_break_disappointment), ";\n",
    "total_preference_disappointment = ", show(total_preference_disappointment), ";\n",
    "objective = ", show(objective), ";\n"
];

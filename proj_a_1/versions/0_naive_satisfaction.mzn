% =============================================================================
% VERSION 0: Naive Satisfaction Solution
% =============================================================================
% Najprostsze możliwe rozwiązanie - tylko spełnienie podstawowych ograniczeń
% bez optymalizacji. Cel: zrozumieć strukturę problemu.
%
% To rozwiązanie:
% - Przypisuje studentów do grup
% - NIE respektuje wykluczeń (-1)
% - NIE respektuje konfliktów między grupami
% - NIE liczy poprawnie funkcji celu
% - Po prostu znajduje JAKIEKOLWIEK przypisanie
% =============================================================================

% Parametry (z enroll.mzn)
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
% ZMIENNE DECYZYJNE
% =============================================================================

% Najprostsza reprezentacja: zbiór grup dla każdego studenta
array[Student] of var set of Group: assignment;

% =============================================================================
% OGRANICZENIA (minimalne)
% =============================================================================

% Dla każdego studenta i każdej klasy - dokładnie jedna grupa z tej klasy
% (lub zero jeśli student nie uczęszcza na tę klasę - ale tu ignorujemy)
constraint forall(s in Student, c in Class) (
    card(assignment[s] intersect {g | g in Group where group_class[g] = c}) <= 1
);

% Każdy student musi mieć przypisane COKOLWIEK (naiwne - co najmniej 1 grupę)
constraint forall(s in Student) (
    card(assignment[s]) >= 1
);

% =============================================================================
% OBLICZENIA DLA WYJŚCIA (naiwne - wszystko = 0)
% =============================================================================

var int: total_break_disappointment = 0;
var int: total_preference_disappointment = 0;
var int: objective = 0;

% =============================================================================
% ROZWIĄZANIE
% =============================================================================

solve satisfy;

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

% =============================================================================
% VERSION 1: Basic Constraints
% =============================================================================
% Dodajemy wszystkie TWARDE ograniczenia:
% - Wykluczenia (exclusion = -1)
% - Konflikty między grupami
% - Limity wielkości grup
% - Dokładnie jedna grupa z każdej klasy (lub zero jeśli wszystkie wykluczone)
%
% Funkcja celu nadal uproszczona - tylko satisfaction.
% =============================================================================

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
% POMOCNICZE ZBIORY
% =============================================================================

% Grupy należące do danej klasy
array[Class] of set of Group: class_groups =
    [{g | g in Group where group_class[g] = c} | c in Class];

% Czy student uczęszcza na daną klasę (przynajmniej jedna grupa nie jest wykluczona)
array[Student, Class] of bool: student_attends_class =
    array2d(Student, Class, [
        exists(g in class_groups[c])(student_prefers[s,g] != exclusion)
    | s in Student, c in Class]);

% Dozwolone grupy dla studenta w danej klasie
array[Student, Class] of set of Group: allowed_groups =
    array2d(Student, Class, [
        {g | g in class_groups[c] where student_prefers[s,g] != exclusion}
    | s in Student, c in Class]);

% =============================================================================
% ZMIENNE DECYZYJNE
% =============================================================================

% Zbiór grup przypisanych do każdego studenta
array[Student] of var set of Group: assignment;

% =============================================================================
% OGRANICZENIA TWARDE
% =============================================================================

% 1. Dla każdego studenta i każdej klasy:
%    - Dokładnie jedna grupa jeśli student uczęszcza na klasę
%    - Zero grup jeśli nie uczęszcza
constraint forall(s in Student, c in Class) (
    if student_attends_class[s,c] then
        % Dokładnie jedna grupa z tej klasy
        card(assignment[s] intersect class_groups[c]) = 1
    else
        % Żadna grupa z tej klasy
        card(assignment[s] intersect class_groups[c]) = 0
    endif
);

% 2. Respektuj wykluczenia - student nie może być przypisany do wykluczonej grupy
constraint forall(s in Student, g in Group) (
    student_prefers[s,g] = exclusion -> not(g in assignment[s])
);

% 3. Respektuj konflikty - student nie może mieć dwóch konfliktujących grup
constraint forall(s in Student, g1 in Group, g2 in Group where g1 < g2) (
    groups_conflicts[g1,g2] -> not(g1 in assignment[s] /\ g2 in assignment[s])
);

% 4. Limity wielkości grup
constraint forall(g in Group) (
    sum(s in Student)(bool2int(g in assignment[s])) <= class_size[group_class[g]]
);

% =============================================================================
% OBLICZENIA DLA WYJŚCIA (placeholder - niepoprawne)
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

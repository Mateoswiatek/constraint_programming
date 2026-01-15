% =============================================================================
% VERSION 2: Correct Objective Calculation
% =============================================================================
% Dodajemy poprawne obliczenie funkcji celu:
% - preference_disappointment: różnica między najlepszą a przypisaną grupą
% - break_disappointment: czas marnowany na przerwy (znormalizowany do godzin)
% - objective: suma kwadratów ważonych rozczarowań
%
% Nadal tylko satisfy (bez optymalizacji).
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
% POMOCNICZE ZBIORY I TABLICE
% =============================================================================

% Grupy należące do danej klasy
array[Class] of set of Group: class_groups =
    [{g | g in Group where group_class[g] = c} | c in Class];

% Czy student uczęszcza na daną klasę
array[Student, Class] of bool: student_attends_class =
    array2d(Student, Class, [
        exists(g in class_groups[c])(student_prefers[s,g] != exclusion)
    | s in Student, c in Class]);

% Dozwolone grupy dla studenta w danej klasie
array[Student, Class] of set of Group: allowed_groups =
    array2d(Student, Class, [
        {g | g in class_groups[c] where student_prefers[s,g] != exclusion}
    | s in Student, c in Class]);

% Najlepsza preferencja studenta dla klasy (max wśród dozwolonych grup)
array[Student, Class] of int: best_preference =
    array2d(Student, Class, [
        if student_attends_class[s,c] then
            max([student_prefers[s,g] | g in allowed_groups[s,c]])
        else
            0
        endif
    | s in Student, c in Class]);

% Koniec grupy (start + duration)
array[Group] of Time: group_end = [group_start[g] + class_duration[group_class[g]] | g in Group];

% =============================================================================
% ZMIENNE DECYZYJNE
% =============================================================================

% Zbiór grup przypisanych do każdego studenta
array[Student] of var set of Group: assignment;

% =============================================================================
% OGRANICZENIA TWARDE
% =============================================================================

% 1. Dla każdego studenta i każdej klasy: dokładnie jedna lub zero grup
constraint forall(s in Student, c in Class) (
    if student_attends_class[s,c] then
        card(assignment[s] intersect class_groups[c]) = 1
    else
        card(assignment[s] intersect class_groups[c]) = 0
    endif
);

% 2. Respektuj wykluczenia
constraint forall(s in Student, g in Group) (
    student_prefers[s,g] = exclusion -> not(g in assignment[s])
);

% 3. Respektuj konflikty
constraint forall(s in Student, g1 in Group, g2 in Group where g1 < g2) (
    groups_conflicts[g1,g2] -> not(g1 in assignment[s] /\ g2 in assignment[s])
);

% 4. Limity wielkości grup
constraint forall(g in Group) (
    sum(s in Student)(bool2int(g in assignment[s])) <= class_size[group_class[g]]
);

% =============================================================================
% OBLICZENIE PREFERENCE DISAPPOINTMENT
% =============================================================================

% Preferencja przypisanej grupy dla studenta w klasie
array[Student, Class] of var int: assigned_preference;

constraint forall(s in Student, c in Class) (
    if student_attends_class[s,c] then
        assigned_preference[s,c] = sum(g in allowed_groups[s,c])(
            bool2int(g in assignment[s]) * student_prefers[s,g]
        )
    else
        assigned_preference[s,c] = 0
    endif
);

% Rozczarowanie preferencyjne dla studenta w klasie
array[Student, Class] of var int: pref_disappointment_per_class;

constraint forall(s in Student, c in Class) (
    if student_attends_class[s,c] then
        pref_disappointment_per_class[s,c] = best_preference[s,c] - assigned_preference[s,c]
    else
        pref_disappointment_per_class[s,c] = 0
    endif
);

% Całkowite rozczarowanie preferencyjne studenta
array[Student] of var int: preference_disappointment;

constraint forall(s in Student) (
    preference_disappointment[s] = sum(c in Class)(pref_disappointment_per_class[s,c])
);

% =============================================================================
% OBLICZENIE BREAK DISAPPOINTMENT
% =============================================================================

% Czas rozpoczęcia pierwszej grupy studenta w danym dniu
array[Student, Day] of var Time: first_start;
array[Student, Day] of var Time: last_end;

% Czy student ma zajęcia w danym dniu
array[Student, Day] of var bool: has_classes_on_day;

constraint forall(s in Student, d in Day) (
    has_classes_on_day[s,d] = exists(g in Group where group_day[g] = d)(g in assignment[s])
);

constraint forall(s in Student, d in Day) (
    if has_classes_on_day[s,d] then
        first_start[s,d] = min(g in Group where group_day[g] = d)(
            if g in assignment[s] then group_start[g] else max(Time) endif
        )
    else
        first_start[s,d] = 0
    endif
);

constraint forall(s in Student, d in Day) (
    if has_classes_on_day[s,d] then
        last_end[s,d] = max(g in Group where group_day[g] = d)(
            if g in assignment[s] then group_end[g] else 0 endif
        )
    else
        last_end[s,d] = 0
    endif
);

% Czas spędzony na uczelni w danym dniu
array[Student, Day] of var int: time_at_uni;

constraint forall(s in Student, d in Day) (
    if has_classes_on_day[s,d] then
        time_at_uni[s,d] = last_end[s,d] - first_start[s,d]
    else
        time_at_uni[s,d] = 0
    endif
);

% Całkowity czas spędzony na uczelni
array[Student] of var int: total_time_at_uni;

constraint forall(s in Student) (
    total_time_at_uni[s] = sum(d in Day)(time_at_uni[s,d])
);

% Minimalny wymagany czas (suma czasów trwania wszystkich klas studenta)
array[Student] of int: min_required_time = [
    sum(c in Class where student_attends_class[s,c])(class_duration[c])
| s in Student];

% Zmarnowany czas (różnica, minimum 0)
array[Student] of var int: wasted_time;

constraint forall(s in Student) (
    wasted_time[s] = max(0, total_time_at_uni[s] - min_required_time[s])
);

% Rozczarowanie przerwowe (znormalizowane do godzin - zaokrąglone w górę)
array[Student] of var int: break_disappointment;

constraint forall(s in Student) (
    break_disappointment[s] = ceil_div(wasted_time[s], n_time_units_in_hour)
);

% =============================================================================
% OBLICZENIE FUNKCJI CELU
% =============================================================================

% Total disappointment dla każdego studenta (ważona średnia)
array[Student] of var int: total_disappointment;

constraint forall(s in Student) (
    total_disappointment[s] = ceil_div(
        student_break_importance[s] * break_disappointment[s] +
        (10 - student_break_importance[s]) * preference_disappointment[s],
        10
    )
);

% Suma kwadratów total disappointment
var int: objective = sum(s in Student)(total_disappointment[s] * total_disappointment[s]);

% Sumy składowych (do wyjścia)
var int: total_break_disappointment = sum(s in Student)(break_disappointment[s]);
var int: total_preference_disappointment = sum(s in Student)(preference_disappointment[s]);

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

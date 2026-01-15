# Version 5: Advanced Search Strategies

## Cel
Zaawansowane strategie przeszukiwania i redundantne ograniczenia.

## Co nowego względem v4

### 1. Heurystyka trudności studentów
```minizinc
array[Student] of int: student_flexibility = [
    sum(c in Class)(card(allowed_groups[s,c]))
| s in Student];

array[int] of Student: students_by_difficulty =
    sort_by(Student, [student_flexibility[s] | s in Student]);
```
Studenci z mniejszą liczbą opcji są przydzielani najpierw.

### 2. Zmiana strategii przeszukiwania
```minizinc
solve :: int_search(
    ordered_vars,
    dom_w_deg,  % zamiast first_fail
    indomain_min
) minimize objective;
```

`dom_w_deg` (domain size / weighted degree) jest często lepsza niż `first_fail`:
- Uwzględnia nie tylko rozmiar dziedziny
- Ale też "aktywność" zmiennej w ograniczeniach

### 3. Redundantne ograniczenia
```minizinc
constraint redundant_constraint(
    forall(c in Class) (
        sum(g in class_groups[c])(group_count[g]) =
        sum(s in Student)(bool2int(student_attends_class[s,c]))
    )
);
```
Suma studentów w grupach = liczba uczęszczających (oczywiste, ale pomaga propagacji).

### 4. Jawne liczniki grup
```minizinc
array[Group] of var 0..n_students: group_count;

constraint forall(g in Group) (
    group_count[g] = sum(s in Student, c in Class where g in class_groups[c])(
        bool2int(student_group[s,c] = g)
    )
);
```
Umożliwia lepszą propagację limitów grup.

## Problemy pozostałe
- `indomain_min` nadal nie uwzględnia preferencji bezpośrednio
- Brak LNS dla bardzo dużych instancji

## Następne kroki (Version 6)
- Implementacja sortowania wartości wg preferencji
- Lub sequential search z różnymi strategiami

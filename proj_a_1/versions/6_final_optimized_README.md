# Version 6: Final Optimized Solution

## Cel
Finalna, w pełni zoptymalizowana wersja modelu.

## Co nowego względem v5

### 1. Zoptymalizowane ograniczenie konfliktów
```minizinc
constraint forall(s in Student) (
    forall(c1 in Class, c2 in Class where c1 < c2 /\
           student_attends_class[s,c1] /\ student_attends_class[s,c2]) (
        forall(g1 in allowed_groups[s,c1], g2 in allowed_groups[s,c2]
               where groups_conflicts[g1,g2]) (
            not(student_group[s,c1] = g1 /\ student_group[s,c2] = g2)
        )
    )
);
```
Iterujemy tylko przez **dozwolone** grupy, nie wszystkie.

### 2. Efektywniejsze liczenie grup
```minizinc
constraint forall(g in Group) (
    group_count[g] = sum(s in Student)(
        bool2int(student_group[s,group_class[g]] = g)
    )
);
```
Używamy `group_class[g]` bezpośrednio.

### 3. Dodatkowe redundantne ograniczenie
```minizinc
constraint redundant_constraint(
    forall(s in Student) (
        sum(c in Class)(bool2int(student_group[s,c] != 0)) =
        sum(c in Class)(bool2int(student_attends_class[s,c]))
    )
);
```
Każdy student ma dokładnie tyle grup, ile klas uczęszcza.

### 4. Sequential search z indomain_split
```minizinc
solve :: seq_search([
    int_search(ordered_vars, dom_w_deg, indomain_split),
    int_search([objective], input_order, indomain_min)
]) minimize objective;
```

`indomain_split` wykonuje binary search na dziedzinie, co często jest szybsze niż `indomain_min`.

## Wszystkie optymalizacje w tym modelu

1. **Reprezentacja**: Zmienne całkowite zamiast zbiorów
2. **Ograniczenie dziedziny**: Od razu do dozwolonych grup
3. **Sortowanie studentów**: Po trudności (mniej opcji najpierw)
4. **Strategia wyboru zmiennych**: `dom_w_deg`
5. **Strategia wartości**: `indomain_split`
6. **Redundantne ograniczenia**: 2 dodatkowe dla propagacji
7. **Zoptymalizowane konflikty**: Tylko dozwolone pary

## Uruchomienie

```bash
minizinc --solver Gecode 6_final_optimized.mzn data/trivial.dzn

# Dla dużych instancji z LNS (Gecode):
minizinc --solver Gecode --restart-base 1.5 --restart constant 6_final_optimized.mzn data/competition.dzn

# Z limitem czasu:
minizinc --solver Gecode --time-limit 60000 6_final_optimized.mzn data/competition.dzn
```

## Różne solvery

Warto przetestować z różnymi solverami:
- **Gecode**: Dobry ogólny solver
- **Chuffed**: Lazy clause generation
- **OR-Tools CP-SAT**: Często najszybszy dla dużych problemów

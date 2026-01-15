# Research Resources - MiniZinc Optimization for Assignment Problems

## Oficjalna Dokumentacja MiniZinc

1. **MiniZinc Handbook - Effective Modeling Practices**
   - https://docs.minizinc.dev/en/stable/efficient.html
   - Najważniejsze praktyki modelowania, bounded variables, redundant constraints

2. **MiniZinc Handbook - Search Annotations**
   - https://docs.minizinc.dev/en/stable/mzn_search.html
   - Strategie przeszukiwania: first_fail, dom_w_deg, impact, indomain_min/max

3. **MiniZinc Global Constraints Library**
   - https://docs.minizinc.dev/en/stable/lib-globals.html
   - alldifferent, inverse, bin_packing, global_cardinality, value_precede

4. **MiniZinc Tutorial (University of Glasgow)**
   - https://www.dcs.gla.ac.uk/~pat/cpM/minizincCPM/tutorial/minizinc-tute.pdf
   - Kompletny tutorial z przykładami

## Akademickie Publikacje

5. **Automated Large-scale Class Scheduling in MiniZinc**
   - https://arxiv.org/pdf/2011.07507
   - Bezpośrednio związane z planowaniem zajęć na uczelni

6. **Symmetry Declarations for MiniZinc**
   - https://people.eng.unimelb.edu.au/pstuckey/minisym/paper.pdf
   - Techniki łamania symetrii (symmetry breaking)

7. **Solver-Independent Large Neighbourhood Search (LNS)**
   - https://people.eng.unimelb.edu.au/pstuckey/papers/MiniLNS.pdf
   - Zaawansowana technika optymalizacji dla dużych problemów

8. **Assignment Problem with Conflicts**
   - https://arxiv.org/html/2506.04274
   - Problem przypisania z konfliktami - teoretyczne podstawy

## Narzędzia i Frameworki

9. **MiniBrass - Soft Constraints for MiniZinc**
   - https://github.com/isse-augsburg/minibrass
   - Framework do modelowania miękkich ograniczeń

10. **Google OR-Tools Constraint Optimization**
    - https://developers.google.com/optimization/cp/
    - Alternatywny solver z dobrą dokumentacją

11. **MiniZinc Challenge 2024 Rules**
    - https://www.minizinc.org/challenge/2024/rules/
    - Porównanie solverów, benchmarki

## Praktyczne Porady (Blogi, Stack Overflow)

12. **Optimizing MiniZinc - Hillel Wayne**
    - https://www.hillelwayne.com/post/minizinc-2/
    - Praktyczne wskazówki optymalizacji, porównanie solverów

13. **Stack Overflow - Multiple Objectives in MiniZinc**
    - https://stackoverflow.com/questions/65091846/optimise-multiple-objectives-in-minizinc
    - Wielokryterialna optymalizacja

14. **Stack Overflow - Custom Search Heuristics in Gecode**
    - https://stackoverflow.com/questions/74594132/minizinc-how-to-implement-custom-search-heuristics-in-gecode
    - Implementacja LNS z Gecode

15. **MiniZinc Google Group - Assignment to Groups**
    - https://groups.google.com/g/minizinc/c/1DkGPUXocUo
    - Dyskusja o problemach przypisania do grup

## Kluczowe Techniki do Zastosowania

### 1. Global Constraints (zamiast ręcznych pętli)
- `global_cardinality` - do limitów grup
- `bin_packing_capa` - alternatywa dla limitów pojemności
- `alldifferent` - jeśli potrzebne unikalne przypisania

### 2. Strategie Przeszukiwania
```minizinc
solve :: int_search(vars, dom_w_deg, indomain_min) minimize objective;
```
- `first_fail` - najmniejsza dziedzina najpierw
- `dom_w_deg` - dziedzina / stopień ważony (często najlepszy)
- `impact` - bazuje na historii wpływu

### 3. Symmetry Breaking
```minizinc
include "lex_lesseq.mzn";
constraint lex_lesseq(row1, row2);
```

### 4. Large Neighborhood Search (LNS)
```minizinc
solve :: relax_and_reconstruct(vars, 90) minimize objective;
```
Uruchomienie: `minizinc --solver Gecode --restart-base 1.5 --restart constant`

### 5. Redundant Constraints
```minizinc
constraint redundant_constraint(sum(...) = expected_total);
```

## Polecane Solvery

1. **Gecode** - dobry ogólny solver, wspiera LNS
2. **Chuffed** - lazy clause generation, dobry dla trudnych problemów
3. **OR-Tools CP-SAT** - doskonały dla dużych instancji
4. **COIN-BC** - dobry dla problemów z liniową strukturą

## Uwagi do Problemu Group Enroll

- Funkcja celu: minimalizacja sumy kwadratów rozczarowań
- Kwadraty penalizują duże rozczarowania bardziej (fairness)
- Warto pre-computować tablicę kwadratów dla wydajności
- Ważone połączenie break_disappointment i preference_disappointment
- ceil_div już zdefiniowane w enroll.mzn

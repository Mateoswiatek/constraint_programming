# Version 3: Basic Optimization

## Cel
Przejście z `solve satisfy` na `solve minimize objective`.

## Co nowego względem v2

### Optymalizacja
```minizinc
solve minimize objective;
```

### Include globals
```minizinc
include "globals.mzn";
```
Daje dostęp do globalnych ograniczeń (przyda się w kolejnych wersjach).

## Zachowanie
- Solver będzie szukał rozwiązania minimalizującego sumę kwadratów
- Domyślna strategia przeszukiwania (zależna od solvera)
- Może być wolne dla dużych instancji

## Problemy pozostałe
- Brak jawnej strategii przeszukiwania
- Reprezentacja jako zbiory może być nieefektywna
- Brak redundantnych ograniczeń dla lepszej propagacji

## Następne kroki (Version 4)
- Zmiana reprezentacji na tablicę zmiennych całkowitych
- Dodanie jawnej strategii przeszukiwania (search annotation)
- Użycie global_cardinality dla limitów grup

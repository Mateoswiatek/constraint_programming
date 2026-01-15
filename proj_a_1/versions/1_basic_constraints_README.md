# Version 1: Basic Constraints

## Cel
Dodanie wszystkich twardych ograniczeń problemu.

## Co nowego względem v0
1. **Wykluczenia**: Student nie może być przypisany do grupy z preferencją -1
2. **Konflikty**: Student nie może mieć dwóch grup, które ze sobą kolidują
3. **Limity grup**: Każda grupa ma maksymalną pojemność (class_size)
4. **Poprawne przypisanie**:
   - Dokładnie 1 grupa z każdej klasy, na którą student uczęszcza
   - 0 grup z klasy, jeśli wszystkie grupy są wykluczone

## Kluczowe struktury pomocnicze
```minizinc
% Grupy należące do klasy
array[Class] of set of Group: class_groups

% Czy student uczęszcza na klasę
array[Student, Class] of bool: student_attends_class

% Dozwolone grupy dla studenta
array[Student, Class] of set of Group: allowed_groups
```

## Problemy pozostałe
- Funkcja celu nadal = 0 (nie optymalizujemy)
- Nie liczymy break_disappointment
- Nie liczymy preference_disappointment

## Następne kroki (Version 2)
- Obliczyć preference_disappointment
- Obliczyć break_disappointment
- Obliczyć poprawną funkcję celu (suma kwadratów)

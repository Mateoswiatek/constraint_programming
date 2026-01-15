# Version 4: Integer Representation with Search Heuristics

## Cel
Zmiana reprezentacji na bardziej efektywną (zmienne całkowite zamiast zbiorów).

## Co nowego względem v3

### Nowa reprezentacja
Zamiast:
```minizinc
array[Student] of var set of Group: assignment;
```

Używamy:
```minizinc
array[Student, Class] of var 0..n_groups: student_group;
```

gdzie `student_group[s,c]` to numer grupy przypisanej studentowi `s` w klasie `c`.

### Zalety nowej reprezentacji
1. **Ograniczenie dziedziny** - od razu do dozwolonych grup:
   ```minizinc
   student_group[s,c] in allowed_groups[s,c]
   ```

2. **Łatwiejsze indeksowanie** - dostęp do preferencji:
   ```minizinc
   assigned_preference[s,c] = student_prefers[s, student_group[s,c]]
   ```

3. **Search annotations** - możliwość użycia strategii:
   ```minizinc
   solve :: int_search(flat_student_group, first_fail, indomain_min)
   ```

### Strategia przeszukiwania
- `first_fail` - wybierz zmienną z najmniejszą dziedziną
- `indomain_min` - przypisz najmniejszą wartość (grupę o niższym indeksie)

### Zbiory dla wyjścia
Budowane z reprezentacji całkowitoliczbowej:
```minizinc
assignment[s] = {g | c in Class, g in class_groups[c] where student_group[s,c] = g}
```

## Problemy pozostałe
- `indomain_min` nie uwzględnia preferencji (przypisuje po indeksie)
- Brak redundantnych ograniczeń
- Brak advanced search (dom_w_deg, LNS)

## Następne kroki (Version 5)
- Dodanie custom value selection opartego na preferencjach
- Redundantne ograniczenia dla lepszej propagacji
- Bardziej zaawansowane strategie przeszukiwania

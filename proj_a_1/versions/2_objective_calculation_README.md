# Version 2: Correct Objective Calculation

## Cel
Poprawne obliczenie funkcji celu zgodnie ze specyfikacją.

## Co nowego względem v1

### Preference Disappointment
Dla każdego studenta i klasy:
```
pref_disappointment = best_preference - assigned_preference
```
gdzie `best_preference` to maksymalna preferencja wśród dozwolonych grup.

### Break Disappointment
1. Dla każdego dnia obliczamy:
   - `first_start` - początek pierwszej grupy
   - `last_end` - koniec ostatniej grupy
   - `time_at_uni` = `last_end - first_start`

2. Całkowity czas na uczelni: `total_time_at_uni = sum(time_at_uni)`

3. Minimalny wymagany czas: suma duration klas studenta

4. Zmarnowany czas: `max(0, total_time_at_uni - min_required_time)`

5. Break disappointment (znormalizowany do godzin):
   ```
   break_disappointment = ceil_div(wasted_time, n_time_units_in_hour)
   ```

### Total Disappointment
Ważona średnia:
```
total_disappointment[s] = ceil_div(
    break_importance[s] * break_disappointment[s] +
    (10 - break_importance[s]) * preference_disappointment[s],
    10
)
```

### Objective
```
objective = sum(s in Student)(total_disappointment[s]²)
```

## Kluczowe zmienne pomocnicze
- `assigned_preference[s,c]` - preferencja przypisanej grupy
- `time_at_uni[s,d]` - czas na uczelni w danym dniu
- `wasted_time[s]` - zmarnowany czas studenta

## Problemy pozostałe
- Nadal tylko `solve satisfy` (nie optymalizujemy)
- Może znaleźć bardzo złe rozwiązanie

## Następne kroki (Version 3)
- Zmiana na `solve minimize objective`
- Dodanie prostej strategii przeszukiwania

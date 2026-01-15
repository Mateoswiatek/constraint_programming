# Hard Solver - Naprawiona wersja

Kopia solvera z `optTorpedo/hotstart/` z naprawionym błędem walidacji konwertera.

## Znaleziony błąd

### Problem: Niespójność czasu konwertera między schedulerem a walidatorem

**Kontekst:**
Torpeda może PRZYJECHAĆ do konwertera (`start_converter`) PRZED jego otwarciem (`conv_opens`).
Rzeczywista konwersja zaczyna się dopiero gdy konwerter się otworzy:
```
actual_conv_start = max(start_converter, conv_opens)
```

**Błąd w oryginalnym kodzie:**

1. **Scheduler** (`_schedule_with_segments`) - poprawnie używał `actual_conv_start`:
```python
# linie ~669-673 w oryginalnym solver.py
actual_conv_start = max(cycle.start_converter, conv_time)
converter_events.append((actual_conv_start, 1))
converter_events.append((cycle.end_converter, -1))
```

2. **Walidator** (`SolutionValidator._check_cumulative`) - błędnie używał `start_converter`:
```python
# linie ~804-809 w oryginalnym solver.py
self._check_cumulative(
    cycles,
    "Converter",
    d.slots_converter,
    lambda c: (c.start_converter, c.end_converter) if c.conv_idx >= 0 else None,  # BUG!
)
```

3. **Funkcja oceny** (`_count_constraint_violations`) - też błędnie używała `start_converter`:
```python
# linie ~1117-1120 w oryginalnym solver.py
conv_intervals = [
    (c.start_converter, c.end_converter) for c in self.cycles if c.conv_idx >= 0  # BUG!
]
```

**Skutek:**
- Walidator zgłaszał naruszenia pojemności konwertera (np. 4-5 torped przy pojemności 3)
- Ale te naruszenia były fałszywe - torpedy tylko CZEKAŁY przy konwerterze, nie używały go jednocześnie
- Solver "utknął" bo myślał że rozwiązanie jest niepoprawne

## Naprawa

### 1. Nowa metoda walidacji konwertera

```python
def _check_cumulative_converter(self, cycles, capacity):
    """Check converter cumulative constraint using actual conversion start time."""
    d = self.data
    events = []
    for c in cycles:
        if c.conv_idx >= 0:
            # Actual conversion starts at max(arrival, converter_opens)
            actual_conv_start = max(c.start_converter, d.conv_opens[c.conv_idx])
            events.append((actual_conv_start, 1, c.pouring_idx))
            events.append((c.end_converter, -1, c.pouring_idx))
    # ... reszta walidacji
```

### 2. Poprawiona funkcja oceny

```python
# W _count_constraint_violations():
conv_intervals = []
for c in self.cycles:
    if c.conv_idx >= 0:
        actual_conv_start = max(c.start_converter, d.conv_opens[c.conv_idx])
        conv_intervals.append((actual_conv_start, c.end_converter))
```

## Gdzie jeszcze może być ten błąd

Sprawdź te miejsca w innych solverach:

1. **Każde miejsce gdzie sprawdzana jest pojemność konwertera** - upewnij się że używa `actual_conv_start`
2. **Funkcje walidacji** - `validate_solution.py` w innych folderach
3. **Funkcje oceny/kosztu** - jeśli liczą naruszenia ograniczeń konwertera
4. **Schedulery** - sprawdź czy są spójne z walidatorami

### Pliki do sprawdzenia:
- `optTorpedo/hotstart/solver.py` - BŁĘDNY (oryginał)
- `optTorpedo/hotstart/validate_solution.py` - OK (używa actual_conv_start)
- Inne solvery w projekcie...

## Użycie

```bash
python run_full.py ../../data/competition.json hard_.sol 300 --checkpoint-interval 60
```

## Dodatkowe poprawki (po odrzuceniu przez competition)

### Problem 2: Ścisłe ograniczenia czasowe

Competition validator wymaga **dokładnej** równości dla pewnych czasów:

1. **startFurnace = metal_start** (EXACT)
   - Oryginalny scheduler próbował opóźniać seg1 w przypadku konfliktów
   - Ale specyfikacja wymaga: torpeda MUSI być przy piecu dokładnie w `metal_start`

2. **startTrip = startFurnace - t_seg1** (EXACT)
   - Wynika z powyższego

3. **endConverter = actual_conv_start + t_convert** (EXACT)
   - Nie można przedłużać konwersji

4. **endTrip = endConverter + t_seg5** (EXACT)
   - Nie można czekać przy konwerterze po zakończeniu

### Problem 3: Konflikty Seg5 NoOverlap

**Problem:**
Cykle były schedulowane w kolejności `metal_start`, ale seg5 zależy od `end_converter`.
Cykl schedulowany później mógł mieć wcześniejszy `end_converter`, powodując konflikt seg5.

**Rozwiązanie - dwufazowe schedulowanie:**

1. **Faza 1**: Oszacuj `end_converter` dla wszystkich cykli
2. **Faza 2**: Scheduluj w kolejności `end_converter` zamiast `metal_start`

```python
# Phase 1: Estimate timings
for i in all_pourings:
    cycle = self._compute_ideal_timing(i, conv_idx)
    estimates[i] = (cycle.end_converter, conv_idx)

# Phase 2: Schedule in end_converter order
normal_cycles.sort(key=lambda x: x[1])  # sort by estimated end_converter
for pouring_idx, _ in normal_cycles:
    cycle = self._schedule_single_cycle(...)
```

### Problem 4: Post-processing powodował konflikty seg3

Stare rozwiązanie próbowało naprawić seg5 przez przesuwanie `end_full_buffer`,
ale to powodowało kaskadowe konflikty w seg3.

Nowe rozwiązanie używa `_reschedule_with_min_seg5()` która:
- Oblicza minimalny `end_converter` żeby seg5 się nie nakładały
- Przelicza czasy wstecz do `end_full_buffer`
- NIE zmienia `start_full_buffer` ani wcześniejszych czasów

## Wyniki po naprawie

- **Valid: True** (lokalna walidacja)
- **Exact timing: Pass** (wszystkie ścisłe ograniczenia spełnione)
- Missed: 2
- Torpedoes: 12 (po Stage 2 optymalizacji)
- Desulf time: ~15950

# Desulf Optimization Solver

Ten folder zawiera zmodyfikowany solver SA zoptymalizowany pod redukcję `totalDesulfTime`.

## Zmiany względem solver z `/4torpedos`

### 1. Nowy move: `_move_reduce_desulf()`

Dedykowany ruch SA do redukcji czasu desulfuryzacji:

```python
def _move_reduce_desulf(self) -> bool:
    """
    Move: Try to reduce total desulfurization time by swapping pouring-conversion
    assignments to minimize sulfur reduction needs.
    """
```

**Algorytm:**
1. Znajduje pourings z wysokim czasem desulfuryzacji (sulfur_reduction > 0)
2. Dla każdego takiego pouringa szuka alternatywnych konwersji z niższym wymaganym desulfem
3. Sprawdza czy swap jest możliwy (deadline, feasibility)
4. Wykonuje swap jeśli zmniejsza total desulf time

### 2. Adaptive Reheat z eskalacją

Zmieniono mechanizm reheatu z prostego na adaptacyjny z eskalacją:

**Przed:**
- Reheat tylko po 50000 iteracji bez poprawy
- Reheat do `temperature * 100`

**Po (wersja 2):**
- Reheat po **15 sekund** bez poprawy (szybszy)
- LUB po **5000 iteracji** bez poprawy
- **Eskalacja temperatury** przy kolejnych reheatach: x1, x2, x5, x10, x20
- Reset eskalacji przy znalezieniu poprawy

```python
REHEAT_TIME_THRESHOLD = 15.0  # seconds (faster)
REHEAT_ITER_THRESHOLD = 5000  # iterations

# Adaptive escalation multipliers
REHEAT_ESCALATION = [1.0, 2.0, 5.0, 10.0, 20.0]

if should_reheat:
    multiplier = REHEAT_ESCALATION[min(reheat_count, len(REHEAT_ESCALATION) - 1)]
    temperature = initial_temperature * multiplier
    reheat_count += 1
```

**Dlaczego to pomaga:**
- Kolejne reheaty mają coraz wyższą temperaturę
- Wyższa temperatura = większa akceptacja gorszych ruchów = większa eksploracja
- Pozwala "wyskoczyć" z głębszych lokalnych minimów

### 3. Perturbacja przy reheacie

Przy każdym reheacie wykonywane są losowe swapy (5-15), aby "wypchnąć" rozwiązanie z basenu atrakcji:

```python
PERTURB_MIN = 5
PERTURB_MAX = 15

if should_reheat:
    perturb_count = random.randint(PERTURB_MIN, PERTURB_MAX)
    for _ in range(perturb_count):
        # Swap random assigned pouring with random emergency pouring
        ...
```

**Dlaczego to pomaga:**
- Samo podgrzanie temperatury nie zmienia stanu rozwiązania
- SA może wrócić do tego samego lokalnego minimum
- Perturbacja wymusza zmianę stanu, dając szansę na eksplorację innego regionu

### 4. Tabu List

Lista zabronionych rozwiązań (ostatnie 100), aby unikać cyklicznego wracania do tych samych stanów:

```python
TABU_SIZE = 100
tabu_list = deque(maxlen=TABU_SIZE)

# Przy każdym ruchu
assignment_hash = tuple(sorted(self.pouring_to_conv.items()))
is_tabu = assignment_hash in tabu_list

# Akceptuj ruch tylko jeśli:
# - jest poprawa (delta < 0) - zawsze akceptuj
# - LUB nie jest w tabu I przechodzi kryterium SA
if delta < 0:
    accept_move = True
elif not is_tabu and random.random() < math.exp(-delta / temperature):
    accept_move = True
```

**Dlaczego to pomaga:**
- Zapobiega "kręceniu się w kółko" między kilkoma rozwiązaniami
- Wymusza eksplorację nowych stanów
- Poprawa zawsze jest akceptowana (nawet jeśli w tabu)

### 5. Zmodyfikowane parametry SA

| Parametr | Przed | Po | Uzasadnienie |
|----------|-------|-----|--------------|
| `cooling_rate` | 0.9995 | 0.9997 | Wolniejsze chłodzenie = lepsza eksploracja |
| `min_temperature` | 1.0 | 0.1 | Niższa temp = dokładniejsze poszukiwanie |
| `REHEAT_TIME` | 30s | 15s | Szybszy reheat = więcej prób ucieczki |
| `REHEAT_ITER` | 10000 | 5000 | Szybszy reheat iteracyjny |
| Move probability (desulf) | 0% | 30% | Nowy dedykowany ruch |

### 6. Nowe prawdopodobieństwa ruchów

```python
move_probs = [0.15, 0.15, 0.05, 0.10, 0.10, 0.15, 0.30]
# 0: swap_pouring_conversion  15%
# 1: reassign_emergency       15%
# 2: swap_conversions          5%
# 3: shift_buffer_times       10%
# 4: compact_schedule         10%
# 5: reassign_to_better_conv  15%
# 6: reduce_desulf (NEW)      30%
```

### 7. Ulepszone logowanie

Dodano wyświetlanie `totalDesulfTime` w logach postępu:

```python
total_desulf = sum(c.desulf_time for c in self.cycles if c.conv_idx >= 0)
print(f"[{elapsed:.1f}s] #{improvements}: {missed} missed, {violations} violations, {max_conc} torpedoes, {total_desulf} desulf")
```

---

## Teoretyczne uzasadnienie mechanizmów escape

### Problem: Lokalne minimum

SA utknęło przy 4 torpedach, 2265 desulf. Dlaczego?

1. **Basin of attraction** - wiele rozwiązań prowadzi do tego samego minimum
2. **Niska temperatura** - po schłodzeniu SA akceptuje tylko poprawy
3. **Brak dywersyfikacji** - te same ruchy = te same regiony

### Rozwiązanie: Mechanizmy escape

| Mechanizm | Problem | Rozwiązanie |
|-----------|---------|-------------|
| **Adaptive Reheat** | Stała temperatura = stała eksploracja | Eskalacja temp. przy kolejnych reheatach |
| **Perturbacja** | Reheat nie zmienia stanu | Losowe swapy "wypychają" z basenu |
| **Tabu List** | Cykliczne wracanie do stanów | Zabronienie ostatnich N rozwiązań |

### Oczekiwany efekt

```
Reheat #1: temp x1.0, perturbacje -> eksploracja lokalna
Reheat #2: temp x2.0, perturbacje -> szersza eksploracja
Reheat #3: temp x5.0, perturbacje -> agresywna eksploracja
...
```

Każdy kolejny reheat bez poprawy zwiększa "siłę" ucieczki.

---

## Użycie

```bash
# Standardowe uruchomienie (5 minut)
python3 solver.py ../../../data/inst_config3_300_200.json output.sol 300

# Z hot-start z istniejącego rozwiązania
python3 solver.py ../../../data/inst_config3_300_200.json output.sol 600 --hot-start final_4torpedo.sol

# Z checkpointami co 2 minuty
python3 solver.py ../../../data/inst_config3_300_200.json output.sol 600 --hot-start final_4torpedo.sol --checkpoint-interval 120

# Z random restart (eksploruje różne regiony po utknięciu)
python3 solver.py ../../../data/inst_config3_300_200.json output.sol 600 --random-restart

# Z random restart i seedem (reprodukowalny)
python3 solver.py ../../../data/inst_config3_300_200.json output.sol 600 --random-restart --seed 42
```

## Walidacja

```bash
python3 validate_solution.py ../../../data/inst_config3_300_200.json output.sol
```

## Oczekiwane rezultaty

- **Torpedy:** 4 (optymalne, nie da się zejść do 3)
- **Missed conversions:** 0
- **totalDesulfTime:** < 2265 (cel optymalizacji)

## Pliki

- `solver.py` - Zmodyfikowany solver SA z mechanizmami escape
- `final_4torpedo.sol` - Rozwiązanie startowe (4 torpedy, 2295 desulf)
- `validate_solution.py` - Walidator rozwiązań

### 8. Rotujące strategie perturbacji (ZAIMPLEMENTOWANE)

Przy każdym reheacie używana jest inna strategia perturbacji (rotacja):

```python
REHEAT_STRATEGIES = ['random_swaps', 'focus_high_desulf', 'shuffle_emergencies']
current_strategy_idx = 0

if should_reheat:
    strategy = REHEAT_STRATEGIES[current_strategy_idx % len(REHEAT_STRATEGIES)]
    current_strategy_idx += 1
    # Apply strategy-specific perturbation
```

**Strategie:**

1. **`random_swaps`** - Losowe swapy między assigned i emergency pourings
   - Podstawowa dywersyfikacja
   - Szybka i prosta

2. **`focus_high_desulf`** - Skupienie na pourings z najwyższym czasem desulfuryzacji
   - Sortuje pourings po desulf time (malejąco)
   - Zamienia te z najwyższym desulfem na emergency
   - Bezpośrednio celuje w redukcję totalDesulfTime

3. **`shuffle_emergencies`** - Przetasowanie emergency pourings
   - Losowo wybiera które pourings idą do emergency
   - Zmienia strukturę rozwiązania bardziej radykalnie

**Dlaczego to pomaga:**
- Różne strategie eksplorują różne regiony przestrzeni rozwiązań
- `focus_high_desulf` bezpośrednio celuje w optymalizowany parametr
- Rotacja zapobiega "utknięciu" w jednym typie perturbacji

**Przykład logów:**
```
ADAPTIVE REHEAT #1 [random_swaps] -> temp=4.00e+05 (x1.0), 9 perturbations
ADAPTIVE REHEAT #2 [focus_high_desulf] -> temp=8.00e+05 (x2.0), 8 perturbations
ADAPTIVE REHEAT #3 [shuffle_emergencies] -> temp=2.00e+06 (x5.0), 12 perturbations
ADAPTIVE REHEAT #4 [random_swaps] -> temp=4.00e+06 (x10.0), 10 perturbations
...
```

### 9. Random Restart z randomizowaną inicjalizacją (ZAIMPLEMENTOWANE)

Po kilku nieudanych reheatach solver może wykonać pełny restart z **losową inicjalizacją**:

```python
RANDOM_RESTART_THRESHOLD = 5  # Po 5 reheatach bez poprawy

# Flaga --random-restart (domyślnie wyłączona)
parser.add_argument('--random-restart', action='store_true')

if random_restart and reheats_since_improvement >= RANDOM_RESTART_THRESHOLD:
    self._create_initial_solution(randomize=True)  # Losowa inicjalizacja!
    # Reset counters, zachowaj best solution
```

**Randomizowana inicjalizacja `_create_initial_solution(randomize=True)`:**
- Tasuje kolejność konwersji w chunkach (nie ściśle po opening time)
- Wybiera losowo z top-3 kandydatów (nie zawsze najlepszy slack)
- Generuje różne punkty startowe dla eksploracji

**Dlaczego to pomaga:**
- Każdy restart trafia w **inny region** przestrzeni rozwiązań
- Greedy init bez randomizacji zawsze daje to samo rozwiązanie
- Z randomizacją możemy eksplorować różne baseny atrakcji

### 10. Seed dla reprodukowalności (ZAIMPLEMENTOWANE)

Parametr `--seed` pozwala powtórzyć eksperyment:

```python
parser.add_argument('--seed', type=int, default=None)

if seed is not None:
    random.seed(seed)
    print(f"Seed: {seed}")
else:
    print("Seed: None (non-reproducible)")
```

**Użycie:**
```bash
# Reprodukowalny eksperyment
python3 solver.py data.json out.sol 300 --random-restart --seed 42

# Powtórzenie tego samego eksperymentu
python3 solver.py data.json out.sol 300 --random-restart --seed 42  # Identyczny wynik!

# Inny eksperyment
python3 solver.py data.json out.sol 300 --random-restart --seed 123  # Różny wynik

# Bez seeda (domyślne, nie-reprodukowalne)
python3 solver.py data.json out.sol 300 --random-restart  # Losowy
```

### 11. Zwiększone parametry perturbacji (ZAIMPLEMENTOWANE)

```python
# Większa siła perturbacji
PERTURB_MIN = 15  # (było 5)
PERTURB_MAX = 40  # (było 15)

# Większy tabu list
TABU_SIZE = 500  # (było 100)
```

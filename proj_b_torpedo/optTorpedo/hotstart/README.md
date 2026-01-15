# Torpedo Scheduling Solver - Hot Start Version

python3 run_full.py ../../data/inst_config3_300_200.json hot_start_2.sol 1800 --hot-start hot_start_1.sol --checkpoint-interval 120


Wersja z obsługą hot start - możliwość kontynuowania optymalizacji z poprzedniego rozwiązania.

## Pliki

- `solver.py` - Stage 1: Simulated Annealing (z hot start)
- `optimizer_stage2.py` - Stage 2: Interval coloring dla torped
- `run_full.py` - Pełny pipeline
- `validate_solution.py` - Walidator

## Użycie

### Fresh start (od zera)
```bash
python3 run_full.py ../data/inst_config3_300_200.json solution.sol 1800
```

### Hot start (kontynuacja z poprzedniego rozwiązania)
```bash
python3 run_full.py ../data/inst_config3_300_200.json solution.sol 1800 --hot-start previous.sol
```

### Kontynuacja tego samego pliku (np. po 2h chcesz dać kolejne 2h)
```bash
# Pierwsze uruchomienie
python3 run_full.py ../data/inst.json best.sol 7200

# Kontynuacja - używa best.sol jako hot start i nadpisuje go lepszym wynikiem
python3 run_full.py ../data/inst.json best.sol 7200 --hot-start best.sol
```

### Długie uruchomienie z checkpointami
```bash
# Checkpointy co 5 minut (300s) - jeśli przerwie się proces, masz zapisane rozwiązanie
python3 run_full.py ../data/inst.json best.sol 14400 --checkpoint-interval 300
```

### Tylko solver (bez stage2)
```bash
python3 solver.py ../data/inst.json solution.sol 1800 --hot-start previous.sol
```

## Jak to działa

1. **Hot Start** - solver wczytuje `assignedConversion` z poprzedniego rozwiązania i używa go jako punkt startowy zamiast greedy initial solution

2. **Checkpointy** - co `--checkpoint-interval` sekund solver zapisuje najlepsze dotychczas znalezione rozwiązanie

3. **Adaptive Reheating** - jeśli solver utknął (brak poprawy przez 60s), temperatura jest podgrzewana bardziej agresywnie

## Workflow dla długiej optymalizacji

```bash
# Dzień 1: Pierwsze 2h
python3 run_full.py ../data/inst.json best.sol 7200 --checkpoint-interval 300

# Dzień 2: Kontynuacja kolejne 2h
python3 run_full.py ../data/inst.json best.sol 7200 --hot-start best.sol --checkpoint-interval 300

# Dzień 3: Jeszcze 4h w nocy
nohup python3 run_full.py ../data/inst.json best.sol 14400 --hot-start best.sol --checkpoint-interval 600 > log.txt 2>&1 &
```

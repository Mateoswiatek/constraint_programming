# Wyniki Optymalizacji MiniZinc - Group Enrollment

## Baseline
- **Objective: 256419** (5 minut, poprzedni najlepszy wynik)

## Najlepszy znaleziony wynik
- **Objective: 256173**
- **Poprawa: 246 punktów (0.096% lepszy)**

## Konfiguracja zwycięska
```bash
minizinc --solver gecode --time-limit 600000 -s -a \
  --restart geometric --restart-base 100 \
  enroll_best.mzn data/competition.dzn
```

## Testowane konfiguracje (10 minut każda)

| Strategia | Restart Base | Najlepszy Objective | Uwagi |
|-----------|--------------|---------------------|-------|
| geometric | 100 | **256173** | NAJLEPSZY! |
| luby | 100 | 256231 | Dobry |
| luby | 250 | 256985 | OK |
| constant | 500 | 257880 | Słaby |

## Testowane solvery

| Solver | Status | Uwagi |
|--------|--------|-------|
| Gecode | **DZIAŁA** | Najlepszy dla tego problemu, wspiera restarty |
| CP-SAT (OR-Tools) | NIE DZIAŁA | Błędy z modelem |
| Chuffed | NIE DZIAŁA | Błędy z modelem |

## Testowane wersje modelu

| Wersja | Status | Uwagi |
|--------|--------|-------|
| enroll.mzn | Działa | Podstawowa wersja |
| enroll_best.mzn | **DZIAŁA NAJLEPIEJ** | Zoptymalizowana wersja |
| versions/6_final_optimized.mzn | UNKNOWN | Nie znajduje rozwiązania w rozsądnym czasie |
| versions/7_competition_tuned.mzn | UNKNOWN | Nie znajduje rozwiązania w rozsądnym czasie |
| versions/9_minimal_propagation.mzn | UNKNOWN | Nie znajduje rozwiązania w rozsądnym czasie |

## Kluczowe wnioski

1. **Gecode z geometric restart** daje najlepsze wyniki
2. **enroll_best.mzn** jest najlepszą wersją modelu
3. **Restart base 100** optymalny dla tego problemu
4. Wersje z folderu `versions/` nie działają (zbyt wolne lub błędy)

## Polecenie do uruchomienia najlepszej konfiguracji

```bash
# 5 minut
minizinc --solver gecode --time-limit 300000 -s -a \
  --restart geometric --restart-base 100 \
  enroll_best.mzn data/competition.dzn

# 10 minut (lepsze wyniki)
minizinc --solver gecode --time-limit 600000 -s -a \
  --restart geometric --restart-base 100 \
  enroll_best.mzn data/competition.dzn

# 30 minut (najlepsze wyniki)
minizinc --solver gecode --time-limit 1800000 -s -a \
  --restart geometric --restart-base 100 \
  enroll_best.mzn data/competition.dzn
```

## Parametry Gecode restart

- `--restart luby` - Luby sequence restart
- `--restart geometric` - Geometric restart (najlepszy!)
- `--restart constant` - Constant restart
- `--restart-base N` - Bazowa liczba failures przed restartem (100 optymalne)

## Data testów
2024-12-02

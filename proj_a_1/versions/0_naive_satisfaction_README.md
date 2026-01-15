# Version 0: Naive Satisfaction Solution

## Cel
Najprostsze możliwe rozwiązanie - zrozumienie struktury problemu i formatu wyjścia.

## Co robi
- Przypisuje studentów do grup (jako zbiory)
- Wymaga co najmniej jednej grupy na studenta
- Wymaga maksymalnie jednej grupy z każdej klasy

## Czego NIE robi
- Nie respektuje wykluczeń (preference = -1)
- Nie respektuje konfliktów między grupami
- Nie sprawdza limitów wielkości grup
- Nie liczy poprawnie funkcji celu (wszystko = 0)

## Problemy
1. Studenci mogą być przypisani do grup, które wykluczyli
2. Studenci mogą mieć nakładające się grupy
3. Grupy mogą przekroczyć limit pojemności
4. Funkcja celu jest fałszywa (zawsze 0)

## Następne kroki (Version 1)
- Dodać ograniczenie wykluczeń
- Dodać ograniczenie konfliktów
- Dodać ograniczenie pojemności grup

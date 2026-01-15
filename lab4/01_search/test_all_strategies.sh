#!/bin/bash

# Test script for N-Queens with different search strategies

echo "Testing N-Queens with N=8 using different search strategies"
echo "============================================================"

# Strategy 1: input_order, indomain_min
echo -e "\n1. Strategy: input_order, indomain_min"
sed -i 's/solve :: .*/solve :: int_search(rows, input_order, indomain_min, complete) satisfy;/' test_search.mzn
time minizinc --solver Gecode test_search.mzn test_data.dzn --statistics 2>&1 | grep -E "(solutions|failures|propagations|nodes|time)"

# Strategy 2: input_order, indomain_median
echo -e "\n2. Strategy: input_order, indomain_median"
sed -i 's/solve :: .*/solve :: int_search(rows, input_order, indomain_median, complete) satisfy;/' test_search.mzn
time minizinc --solver Gecode test_search.mzn test_data.dzn --statistics 2>&1 | grep -E "(solutions|failures|propagations|nodes|time)"

# Strategy 3: first_fail, indomain_min
echo -e "\n3. Strategy: first_fail, indomain_min"
sed -i 's/solve :: .*/solve :: int_search(rows, first_fail, indomain_min, complete) satisfy;/' test_search.mzn
time minizinc --solver Gecode test_search.mzn test_data.dzn --statistics 2>&1 | grep -E "(solutions|failures|propagations|nodes|time)"

# Strategy 4: first_fail, indomain_median
echo -e "\n4. Strategy: first_fail, indomain_median"
sed -i 's/solve :: .*/solve :: int_search(rows, first_fail, indomain_median, complete) satisfy;/' test_search.mzn
time minizinc --solver Gecode test_search.mzn test_data.dzn --statistics 2>&1 | grep -E "(solutions|failures|propagations|nodes|time)"
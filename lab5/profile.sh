#!/bin/bash
# Requires bash version >= 4 and GNU grep
# mzn2fzn, fzn-gecode should be added to PATH 
# Result: test.log file, where every line says if the test has been passed

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

GREP="grep"
is_grep_gnu=$(grep --version | grep "GNU grep")
if [ -z "$is_grep_gnu" ]; then 
    # on macos ggrep is a popular alias for gnu grep
    GREP="ggrep"
fi
model_basename="portschedule" 
model="$model_basename.mzn"
output_fzn="$model_basename.fzn"
output_ozn="$model_basename.ozn"
test_data="data/ps_04_12_00.dzn"
timelimit=9000
part=228
full=225
# 1, if $full is the optimum
optimum=114
# -lt for maximization, -gt for minimization
comparator="-gt"
solver="fzn-gecode -time $timelimit"

syntax_errors=$(minizinc --instance-check-only $model.mzn $test_data 2>&1)
if [ -z "$syntax_errors" ]; then
    echo "0% - no solution at all, doesn't compile"
    exit
fi

full_result=$(minizinc -c --ozn $output_ozn --solver Gecode $model 2>&1 $test_data && $solver $output_fzn | minizinc --output-time --ozn-file $output_ozn)
    
timeout=$(echo "$full_result" | $GREP "=====UNKNOWN=====")
if [ -n "$timeout" ]; then 
    echo "0% - no solution at all, timeout"
    exit 
fi

unsatisfiable=$(echo "$full_result" | $GREP "=====UNSATISFIABLE=====")
if [ -n "$unsatisfiable" ]; then 
    echo "%0 - no solution at all, solver found problem unsatisfiable"
    exit
fi

result=$(echo "$full_result" | $GREP -Po 'obj\s*=\s*\d+;' | tail -n 1 | $GREP -Po '\d+')
time=$(echo "$full_result" | $GREP -Po '% time elapsed: \d+.\d+ s' | tail -n 1 | cut -d':' -f 2 | tr -d '[:space:]')
echo "$full_result"

if [ -z "$result" ]; then
    echo "0% - output is missing 'obj = <result>;' line"
    exit
fi

found_optimum=$(echo "$full_result" | $GREP "==========")
score="0%"

if [ -n "$found_optimum" ] && [ "$result" -ne "$optimum" ]; then
    score="0% - $result ($time) is not optimum (but the model believes otherwise)"
elif [ $result "$comparator" $part ]; then 
    score="33% - $result ($time) "
elif [ $result $comparator $full ]; then 
    score="66% - $result ($time)"
elif [ $result -eq $full ]; then 
    score="100% - $result ($time)"
elif [ $result $comparator $optimum ] || [ $result -eq $optimum ]; then 
    score="133% - $result is great! ($time)"
else 
    score="0% - $result is impossible"
fi 

echo $score

rm $output_fzn 2>/dev/null || true 
rm $output_ozn 2>/dev/null || true
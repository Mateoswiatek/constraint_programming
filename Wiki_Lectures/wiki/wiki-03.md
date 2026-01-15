# Constraint Programming: Basic Modelling Techniques[](#constraint-programming-basic-modelling-techniques)

## Symmetry Breaking[](#symmetry-breaking)

The problem is said to contain symmetry if there exist classes of equivalent solutions — solutions, which are called symmetrical because there exists a simple mechanical procedure to obtain one from another. Graph Coloring Problem has a very obvious symmetry — in every solution, we can freely swap colors, e.g., every red node repaint as blue, and every blue node repaint as red. Solutions of this kind aren't bad, just redundant, leading to a much bigger search space. Symmetry breaking prunes the search space by removing symmetries from the model.

### Graph Coloring[](#graph-coloring)

  * Problem: Same as [before](https://gitlab.com/agh-courses/25/cp/wiki/02/-/wikis/home "home")
  * Assignment: 
    * Look at and comprehend `01_graph_coloring_symmetry/graph_coloring_symmetry.mzn` model.
    * Try to solve the `basic_data.dzn` instance. 
      * You can use the model created during previous classes
      * There is a chance, that the problem would be too difficult to be solved in a reasonable time.
    * File `data_with_clique.dzn` includes info about the largest clique in the graph 
      * `minColorsNumber` – size of the largest clique
      * `maxClique` – indices of the vertices forming the largest clique
    * Improve model to make use of the info about the largest clique
    * Try to solve the problem again.

### Multi-Knapsack Problem[](#multi-knapsack-problem)

  * Definition: Knapsack problem with several identical knapsacks instead of one.
  * Assignment: 
    * Find symmetry in the model.
    * Look at and comprehend `02_multi_knapsack_symmetry/multi_knapsack.mzn`. 
      * ‼️ Note ‼️ this model is not an example of a good constraint model; it's just used to show a common technique to break symmetries
      * There is a defined predicate named `knapsack` — it's the first example of a user-defined predicate
    * Run model with the associated data file: 
      * The problem may be too hard for the solver
    * Break symmetry using [`lex_less_eq` global constraint](https://www.minizinc.org/doc-latest/en/lib-globals.html#lexicographic-constraints).
    * Run the new model with the same data file

## Redundant Constraints[](#redundant-constraints)

There is a good chance the problem can be defined in more than one way. Furthermore, you may find a set of constraints that is sufficient to define the problem. That's cool. However, there can exist so-called "redundant constraints"; redundant because they do not have an impact on the number or quality of the solutions. The only reason to include them in the model is that they may contain additional info about the structure of the problem, therefore giving the solver an opportunity to prune the search space (most of the solvers prune the search space by propagating constraints, a redundant constraint may boost this process).

### Magic Sequence[](#magic-sequence)

  * Definition: Same as [before](https://gitlab.com/agh-courses/25/cp/wiki/02/-/wikis/home "home")
  * Assignment: 
    * Look at and comprehend `03_magic_sequence_redundant/magic_sequence_redundant.mzn`
    * Add redundant constraints, hints: 
      * what should be equal to the sum of the magic sequence?
      * what should be equal to the sum of sequence elements multiplied by their indexes (if indexing starts from 0)?
    * Compare solving time with and without the redundant constraints.
    * Smile with satisfaction

## Channeling[](#channeling)

If you have more than one model of the same problem, you can combine them into a single model. Why would one do that? Mostly because some constraints are easier to express with different variables. Another reason could be that the second model often makes a great example of redundant constraints.

  * Problem: [N-Queens](https://gitlab.com/agh-courses/25/cp/wiki/02/-/wikis/home "home") again
  * Assignment: 
    * Look at and comprehend `04_n_queens_channeling/n_queens_channeling.mzn`
    * Add another model of the problem 
      * try to use the boolean array of variables `array[1..n, 1..n] of var bool: qb;` (queen boolean)
      * add missing constraints, so the second model was also independent
    * Channel constraints from both models: 
      * create a constraint that connects variables from the model
    * Compare the running time of the normal and channeled model
    * Add symmetry breaking to the problem by using the `lex_lesseq` constraint on the different permutations of the `qb` array 
      * below the assignments there is a code listing with all permutations calculated in MiniZinc, can you tell what symmetries they represent?
    * Compare running time again

Give yourself a [self-five](https://youtu.be/kMUkzWO8viY) in this case, it may not improve the running time, but the technique itself is very useful in more complex problems
```
    array[int] of var bool: qb0 = array1d(qb);
    array[int] of var bool: qb1 = [ qb[j,i] | i,j in 1..n ];
    array[int] of var bool: qb2 = [ qb[i,j] | i in reverse(1..n), j in 1..n ];
    array[int] of var bool: qb3 = [ qb[j,i] | i in 1..n, j in reverse(1..n) ];
    array[int] of var bool: qb4 = [ qb[i,j] | i in 1..n, j in reverse(1..n) ];
    array[int] of var bool: qb5 = [ qb[j,i] | i in reverse(1..n), j in 1..n ];
    array[int] of var bool: qb6 = [ qb[i,j] | i,j in reverse(1..n) ];
    array[int] of var bool: qb7 = [ qb[j,i] | i,j in reverse(1..n) ];
```

## Reified Constraints[](#reified-constraints)

[Reification](https://en.wikipedia.org/wiki/Reification) in Constraint Programming means treating the constraint as a first-order citizen, i.e., you can use the constraint as a boolean value in your model. If you've used the `bool2int` function in the Magic Sequence problem, you could do that only because the constraint `=` has been reified. Reification allows us to create models with "soft constraints" or "conditional constraints", i.e., one constraint has to be satisfied only if the second one is satisfied too, otherwise, they both can be ignored. To achieve that, one has only to reify the constraints and connect them with the implication: `constraint1 -> constraint2`. Let's practice this quite useful technique :)

### Stable Marriage Problem[](#stable-marriage-problem)

  * Problem: There are two classes of objects (men and women, for example) that have to be matched according to their preferences. We say that a matching (marriage) is unstable if both spouses would prefer to be with somebody else. You can read more about this problem on [wikipedia](https://en.wikipedia.org/wiki/Stable_marriage_problem).
  * Assignment: 
    * Look at and comprehend `05_stable_marriage_reification/stable_marriage.mzn`
    * Add missing variables, constraints
    * Give a high-five to your teacher :)

##  🎈 Bonus Stage 🎈[](#balloon-bonus-stage-balloon)

`06_bonus_multi_knapsack_ortools/multi_knapsack.py` contains model of a `multi_knapsack` problem using or-tools CP-SAT solver. There is only a single thing missing — symmetry breaking via `lex_less_eq`. Sadly, CP-SAT does not support this global constraint. Your task is to implement this global constraint using normal constraints. You will have to do manual reification and other crazy things — maybe then you will appreciate MiniZinc? 😁

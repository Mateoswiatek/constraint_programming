# Constraint Programming: Basic Problems[](#constraint-programming-basic-problems)

In this laboratory, we will solve some toy problems from the constraint programming literature. The created models will be later improved using some basic constraint programming techniques.

All files required to solve the assignments are available in the repository, so clone it first.

Protips:

  * If you still have issues with MiniZinc syntax, don't panic! There is a nice cheatsheet built-in into the IDE: `MiniZincIDE -> Help -> Cheat Sheet...`.
  * to get a row/col of a matrix, you can use [`row`](https://www.minizinc.org/doc-2.6.0/en/lib-stdlib.html#index-448)/[`col`](https://www.minizinc.org/doc-2.6.0/en/lib-stdlib.html#index-413) functions
  * MiniZinc supports also [array slicing (ala Python numpy arrays)](https://www.minizinc.org/doc-2.6.0/en/spec.html?highlight=slicing#array-slice-expressions), so you can easily get any sub-array of a given array. So much wow.

## 1\. N-Queens[](#1-n-queens)

  * Definition: you have to place n queens on the chessboard of size `n`x`n` in the way they won't attack each other (see: [wikipedia](https://en.wikipedia.org/wiki/Eight_queens_puzzle)).
  * Assignment: 
    * Fill in the missing code in the corresponding minizinc file.

## 2\. N-Queens + Global Constraints[](#2-n-queens--global-constraints)

One of the simplest (and also the best) methods to write a good model is to make use of the global constraints. The global constraints are different from the normal constraints in two aspects:

  * they may constraint many (more than two) variables at once;
  * they are implemented using very efficient algorithms, which may take advantage of the problem structure in a better way than many binary constraints.

### Connect the dots…[](#connect-the-dots)

  * The `alldifferent` global constraint takes an array of variables as an argument and makes sure that every variable in the array has a different value.
  * Assignment: use the `alldifferent` in the N-Queens model. There is a separate model just for that. Try to replace all the constraints!

To use a global constraint first, you've to find it in [the global constraints' library](https://www.minizinc.org/doc-latest/en/lib-globals.html). Then you've to import in the model, e.g., to use the `cumulative` constraint, you've to add the `include "cumulative.mzn";` line to the model.

## 3\. Graph Coloring[](#3-graph-coloring)

  * Definition: assign colors to the vertices in a graph in the way that no adjacent vertices have the same color. In the case of the planar graph, you will require only four colors (why? there is a very famous [proof of this fact](http://en.wikipedia.org/wiki/Four_color_theorem). Otherwise, you may need even as many colors as there are vertices. More on the [wikipedia page](http://en.wikipedia.org/wiki/Graph_coloring#Vertex_coloring).
  * Assignment: 
    * Fill in the missing code in the corresponding minizinc file.

## 4\. Sudoku[](#4-sudoku)

  * Definition: implement a sudoku solver (see: [Wikipedia](http://en.wikipedia.org/wiki/Sudoku))
  * Assignment: 
    * Fill in missing code in the corresponding minizinc file.

## 5\. Magic Sequence[](#5-magic-sequence)

  * Definition: a sequence `x0, x1, x2,.., xN` is called magic when (for every `i`) `xi` is equal to the number of `i` occurrences in the sequence. An example of the magic sequence with 5 elements is `2, 1, 2, 0, 0` \--- at `0` index you can see how many `0` occur in the sequence, at index `1` how many `1` occur in the sequence, etc.
  * Assignment: 
    * Fill in the missing code in the corresponding minizinc file.


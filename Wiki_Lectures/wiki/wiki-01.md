# Lab 01. XKCD Problem[](#lab-01-xkcd-problem)

Constraint Programming (CP) is a declarative paradigm of programming, which represents a problem by specifying features required from its solutions. The programmer has to:

  * define all variables occurring in the problem along with their domains;
  * define constraints on the introduced variables, which have to be satisfied by the solutions;
  * [optional] define optimality criteria, e.g. cost function, which has to be minimized;
  * [optional] define the search procedure which will be used to find the solution.

## "Real Life" Example[](#real-life-example)

Let's pretend, that we work as a waiter in a restaurant very close to the university campus. One sunny lazy day, there was a computer science conference going on at the campus, and during the lunch break, our restaurant has been attacked by a horde of hungry nerds. One of them was particularly vicious and he gave us a very stupid order:

[![xkcd comic on np-complete problems](68747470733a2f2f696d67732e786b63642e636f6d2f636f6d6963732f6e705f636f6d706c6574652e706e67.png)_http://xkcd.com/287/_](http://xkcd.com/287/)

The situation appeared hopeless as the problem is non-polynomial... Luckily we've heard of constraint programming and without any hesitation started to solve the problem!

### Software[](#software)

  * Install MiniZinc bundle according to the [instructions](http://www.minizinc.org/software.html).
  * Run MiniZinc IDE

## Solving the Problem[](#solving-the-problem)

Modeling problem in MiniZinc consists of four steps:

### 1\. First Step: Variables[](#1-first-step-variables)

Our goal is to complete the insane order. Every order consists of meals, where every meal may be ordered more than once. Let's assume that man can't order half of the meal. Let's define one variable per meal and define their domains as integers. Every variable indicates how many times the meal occurs in the order. In MiniZinc one would write (note the `var` keyword):
```
    var int: fruit;
    var int: fries;
    var int: salad;
    var int: wings;
    var int: sticks;
    var int: sampler;
```

### 2\. Second Step: Constrains[](#2-second-step-constrains)

We can be fairly sure that man can't order a negative number of meals. We can code it in MiniZinc using `constraint` keyword:
```
    constraint fruit >= 0;
    constraint fries >= 0;
    constraint salad >= 0;
    constraint wings >= 0;
    constraint sticks >= 0;
    constraint sampler >= 0;
```

Moreover, we know, that the order has to have a specific price. In order to avoid floats, we will multiply all the prices by 100.
```
    constraint fruit*215 + fries*275 + salad*335 + wings*355 + sticks*420 + sampler*580 == 1505;
```

### 3\. Step Three: Goal[](#3-step-three-goal)

We are looking for an order that **satisfies** the constraints. The `solve` keyword defines the goal.
```
    solve satisfy;
```

### 4\. Output[](#4-output)

Finally we have to define, how to present results to the user. We use the `output` keyword:
```
    output ["fruit:", show(fruit), "\t fries:", show(fries), 
            "\t salad:", show(salad), "\t wings:", show(wings),
            "\t sticks:", show(sticks), "\t sampler:", show(sampler)];
```

#### Running Code[](#running-code)

The simplest way: `MiniZincIDE -> Menu -> Minizinc -> Run`.

To receive more than one solution, check the `Show configuration editor` button in the right top corner, check the `User-defined behavior` checkbox and select `Print all solutions`.

## Tip 50%[](#tip-50)

The comic strip claims, that we will receive a tip only if we manage to create a general solution. We like tips, so let's get to work!

### 1\. First Step: Parameters[](#1-first-step-parameters)

Let's begin with defining the menu's structure, how many meals are in there? Notice the lack of the `var` keywords. It's not a variable, it's a **constant** parameter.
```
    int: menu_length = 6;
```

Then, what is the required price of the order:
```
    int: money_limit = 1505;
```

Next, we will use arrays to describe the menu's contents. Every array has defined a set of indexes and type of the elements:
```
    array[1..menu_length] of int: menu_prices = [215, 275, 335, 355, 420, 580];
    array[1..menu_length] of string: menu_names = ["fruit", "fries", "salad", "wings", "sticks", "sampler"];
```

### 2\. Second Step: Variables[](#2-second-step-variables)

In the general version, there should be as many variables as there is meals in the menu:
```
    array[1..menu_length] of var int: order;
```

### 3\. Third step: Constraints[](#3-third-step-constraints)

Some constraints also depend on the menu's size. To define constraints we will use aggregating functions: `forall` – concatenates all the constraints in a table, `sum` – sums numbers in a table. We will also use an `array comprehension`, which processes the elements of an array according to the defined operation and index set, e.g. `[array[i]*2 | i in 1..array_length]` returns an array with all the values multiplied by `2`. The pipe operator `|` separates the operation and index set we iterate over.
```
    constraint forall([order[i] >= 0 | i in 1..menu_length]);
    constraint sum([order[i] * menu_prices[i] | i in 1..menu_length]) == money_limit;
```

### 4\. Fourth Step: Goal[](#4-fourth-step-goal)

No changes here.

### 5\. Fifth Step: Output[](#5-fifth-step-output)

Again we will use array comprehension. The double-plus operator concatenates strings, while the `show` function converts a number to a string.
```
    output [menu_names[i] ++ ": " ++ show(order[i]) ++ "\n" | i in 1..menu_length];
```

The other, more modern way is to use string interpolation, the line below is equivalent to the one above (notice the `\(...)` notation, things inside such brackets get interpolated into the string):
```
    output ["\(menu_names[i]): \(order[i])\n" | i in 1..menu_length];
```

#### Running Code[](#running-code-1)

No changes here. Please run the code, and spent the imaginary tip wisely.

## Adding a Menu Card[](#adding-a-menu-card)

Mixing parameters and variables is not a good practice – every time the menu changes we would have to modify the program. To solve this issue we can define the parameters' values in the separate data files. Please create a data file (`MiniZincIde -> File -> New data`) with contents:
```
    menu_length = 6;
    money_limit = 1505;
    menu_prices = [215, 275, 335, 355, 420,580];
    menu_names = ["fruit", "fries", "salad", "wings", "sticks", "sampler"];
```

Then please remove the parameters' values from the program. Leave only the empty declarations, e.g. replace `int: menu_length = 6;` with `int: menu_length;`.

#### Running Code[](#running-code-2)

When you hit the `run`, you should be prompted to fill in the missing values. You could do it manually, but I strongly recommend choosing a data file from the drop-down list. The data file has to be opened in the IDE.

## The Wallet Problem[](#the-wallet-problem)

So far we had only to `satisfy` the problem. There exist two different kinds of goals in the MiniZinc language: `maximize <expression>` and `minimize <expression>`, both are used to define the optimality criteria, e.g.
```
    solve minimize sum(order);
```

modifies our problem to look for an order with the lowest number of meals.

### Exercises[](#exercises)

  * Please add another parameter called `yumyum_factors`. It should be an array of integers, which say how much we like a particular meal (the bigger the number is, the more we like the meal).
  * We are searching for a solution, which maximizes the sum of the yumyum factors in our order.
  * We don't have to spend all the money we have.

## Polishing rough edges[](#polishing-rough-edges)

Now we should have a basic working model for the problem at hand, but there are still few ways to improve it. Moreover, this is a good moment to play with the MiniZinc compiler – how to debug and find faults in the model.

### Exercises[](#exercises-1)

  * Can you think of a more succinct way to assert that you can't buy a negative number of meals? Currently we use constraints `>= 0` \- remove them. Tip: `int` can be replaced with any set of integers.
  * We should make sure that our model won't accept illegal parameters. Try special debugging functions to debug your program and assert your requirements. Try functions below, they can be pretty useful in the future: 
    * `abort("Message")`
    * `trace("Message")`
    * `assert(<test>, "Message")`
    * `trace("Message", <expression>)`
    * `assert(<test>, "Message", <expression>) `
    * where: 
      * `<test>` is something like `x < 3`
      * `<expression>` may be a constraint or anything that returns a value
      * you can use double-plus and `show ` to build more complicated messages
  * Ranges, e.g. `1..menu_length` are special cases of `sets`. Replace all the ranges with named sets to avoid code duplication. The declaration should like this:

```
    set of <type>: SetName = start..end;
```

  * `Show configuration editor` is a very tempting button in the right top corner of the MiniZincIDE. Play with the options: 
    * Check the verbose output of the solver.
    * Try using the `Gecode-gist` solver. It should display a search tree. Do you understand what does it show?
    * In case of any issues the current [MiniZinc documentation is here](https://www.minizinc.org/doc-latest/en/index.html)
    * One nice under-documented constant is the 'infinity', which may be used to define unsigned integers and other similar domains
    * Experiment :)

## Assignment[](#assignment)

Fill up the missing code in the `assignment/assignment.mzn` file. The model should include all the improvements/contents you've read about in this document. From the feature-centric perspective, it should contain wallet constraints and the yumyum objective. You are welcome to create your own input data based on the ones already available in the repository.

Finally, don't modify parameters and output sections. It may be useful to understand them though:

  * `index_set(menu_prices)` returns a set containing all indices of the given array, therefore the `menu_length` param becomes obsolete.
  * `output` uses string interpolation and includes all the interesting variables

#  🎈 Extra Fun 🎈[](#balloon-extra-fun-balloon)

In case somebody finished the assignment and was looking for more fun, one can consider how to deploy the Constraint Programming model in production. The company should not be expected to run "MiniZincIDE", etc.

One way is to use an official [MiniZinc Python integration](https://minizinc-python.readthedocs.io/en/latest/). Another way is to forget about the MiniZinc and use a Constraint Programming solver directly. Often, it enables a better integration and allows controlling the solver on a more fine-grained level. Currently, the best free solver is called CP-SAT and is developed as a part of the [Google's OR-Tools project](https://developers.google.com/optimization?hl=pl).

`xkcd.py` file contains a Python code using the CP-SAT directly to solve the XKCD problem. Please don't hesitate to run it and read the source code. Before running, make sure you have a compatible Python version:
```
    # python --version
```

The code requires at least `3.10`. If the requirement is satisfied, create a correct virtual environment:
```
    # python -m venv ortools
    # pip install -r requirements.txt
```

And, finally, run the solver:
```
    # python xkcd.py -a data.json
```

Thanks to `-a` it will print all correct solutions — the results be the same as previously got via MiniZinc.

##  💪 Exercise 💪[](#muscle-exercise-muscle)

Try to extend the `xkcd.py` to handle the yumyum factors, i.e., recreate the assignment in python and ortools. [The official google tutorial](https://developers.google.com/optimization/cp?hl=en) may be helpful.

Just, to make things clear — this will **not** be graded, this exercise is just for your benefit.

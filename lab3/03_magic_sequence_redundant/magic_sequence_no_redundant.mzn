include "distribute.mzn";

int: N;
set of int: Indexes = 0..N-1;

array[Indexes] of var Indexes: magic;

constraint distribute(magic, array1d(Indexes, Indexes), magic);

solve satisfy;

output [ "magic sequence = ", show(magic),";\n"];
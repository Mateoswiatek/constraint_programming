int: menu_length;
int: money_limit;

array[1..menu_length] of int: menu_prices;
array[1..menu_length] of string: menu_names;

% array[1..menu_length] of var int: order;
array[1..menu_length] of var 0..1000: order; % zmienne int-owe muszą mieć ograniczenia

constraint forall(i in 1..menu_length)(order[i] >= 0 /\ order[i] < money_limit div menu_prices[i]);
 
constraint sum([order[i] * menu_prices[i] | i in 1..menu_length]) == money_limit;

solve minimize sum(order);

output ["\(menu_names[i]): \(order[i])\n" | i in 1..menu_length];


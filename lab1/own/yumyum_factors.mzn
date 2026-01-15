int: menu_length;
int: money_limit;
set of int: menuSet = 1..menu_length;
array[menuSet] of int: menu_prices;
array[menuSet] of string: menu_names;

% array[1..menu_length] of var int: order;
array[menuSet] of var 0..1000: order; % zmienne int-owe muszą mieć ograniczenia
array[menuSet] of var 0..1000: yumyum_factors;

constraint forall(i in menuSet)(order[i] >= 0 /\ order[i] < money_limit div menu_prices[i]);
 
constraint sum([order[i] * menu_prices[i] | i in 1..menu_length]) <= money_limit;

solve maximize sum(order);

output ["\(menu_names[i]): \(order[i])\n" | i in 1..menu_length];


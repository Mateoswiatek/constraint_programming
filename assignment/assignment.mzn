% Parameters
int: money_limit;

% notice the "index_set" removed the redundant menu_lenth parameter
set of int: menu_items = index_set(menu_prices);

array[int] of int: menu_prices;
array[menu_items] of string: menu_names;
array[menu_items] of int: yumyum_factors;

% Assertions
constraint assert(money_limit > 0, "money_limit must be positive");
constraint assert(forall(i in menu_items)(menu_prices[i] > 0), "all menu prices must be positive");

% Variables
array[menu_items] of var int: order;
var int: total_price;
var int: total_yumyum;

% Constraints
constraint forall(i in menu_items)(order[i] >= 0 /\ order[i] < money_limit div menu_prices[i]);
constraint total_price = sum([order[i] * menu_prices[i] | i in menu_items]);
constraint total_price == money_limit;
constraint total_yumyum = sum([order[i] * yumyum_factors[i] | i in menu_items]);

% Objective
solve maximize total_yumyum;

output ["\(menu_names[i]): \(order[i])\n" | i in menu_items] ++
       ["- total price:\t\(total_price)/\(money_limit)\n"]
       ++ ["- total yumyum:\t\(total_yumyum)"];

% Parameters
int: money_limit;

% notice the "index_set" removed the redundant menu_lenth parameter
set of int: menu_items = index_set(menu_prices); 

array[int] of int: menu_prices;
array[menu_items] of string: menu_names;
array[menu_items] of int: yumyum_factors; 

% TODO:
% - add assertions/specify domains excluding useless parameters' values
% - add variables
% - add constraints 
% - add objective
% TIP. for missing variables, use names that are already in the output definition. 

output ["\(menu_names[i]): \(order[i])\n" | i in menu_items] ++ 
       ["- total price:\t\(total_price)/\(money_limit)\n"] ++
       ["- total yumyum:\t\(total_yumyum)"];

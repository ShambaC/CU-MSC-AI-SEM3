% Facts
man(marcus).
pompeian(marcus). 
ruler(caesar).
tries_to_assassinate(marcus, caesar).

% Rules
roman(X) :- pompeian(X).
loyal_to_someone(X) :- roman(X).
loyal_to_caesar(X) :- roman(X), \+ tries_to_assassinate(X, caesar).
hate(X, caesar) :- \+ loyal_to_caesar(X).
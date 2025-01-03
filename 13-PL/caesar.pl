% Knowledge base

man(marcus).
pompeian(marcus).
ruler(caesar).

roman(X) :- pompeian(X).
loyal(X, _) :- True.
loyal(X, caesar) :- roman(X), not(hate(X, caesar)).
hate(X, caesar) :- roman(X), not(loyal(X, caesar)).

assassinate(X, Y) :- not(loyal(X, Y)), ruler(Y).
assassinate(marcus, caesar).

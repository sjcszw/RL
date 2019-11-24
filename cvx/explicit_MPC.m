
model = LTISystem('A', [1 1; 0 1], 'B', [0.5; 1]);
model.x.min = [-5; -5];
model.x.max = [5; 5];
model.u.min = -1;
model.u.max = 1;
Q = [1 0; 0 1];
model.x.penalty = QuadFunction(Q);
R = 1;
model.u.penalty = QuadFunction(R);


N = 5;
mpc = MPCController(model, N)

expmpc = mpc.toExplicit();
expmpc.feedback.fplot()
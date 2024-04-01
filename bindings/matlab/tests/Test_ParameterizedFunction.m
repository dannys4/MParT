opts = MapOptions(); %Setting options to default

multis2 = [0 1;2 0];
mset2 = MultiIndexSet(multis2);

ParFunc = CreateComponent(mset2,opts);
ParFunc.SetParams(ones(1,ParFunc.numParams))

disp('Params')
disp(ParFunc.Params)

disp('Evaluate')
disp(ParFunc.Evaluate(randn(2,10)))

tic

load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1637668/data.mat');
Xtrn = single(dataset.train.images);
Ctrn = single(dataset.train.labels);
Xtst = single(dataset.test.images);
threshold = 1;


[Cpreds] = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold);
toc

Ctrues = dataset.test.labels;

N = size(Xtst,1);

[cm, acc] = my_confusion(Ctrues, Cpreds);
Nerr = sum(sum(cm)) - trace(cm);

save('cm.mat', 'cm');

T = table(N, Nerr, acc)

toc
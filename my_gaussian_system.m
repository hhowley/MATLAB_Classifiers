load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1637668/data.mat');

Xtrn = single(dataset.train.images)./255.0;
Ctrn = single(dataset.train.labels);
Xtst = single(dataset.test.images)./255.0;
epsilon = 0.01;
  
tic
[Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon);
toc

Ctrues = dataset.test.labels;

N = size(Xtst,1);

[cm, acc] = my_confusion(Ctrues, Cpreds);
Nerr = sum(sum(cm)) - trace(cm);

save('Task3/cm.mat', 'cm');
mean_jean_walker = Ms(:,26);
clarence_covarience_carlson = Covs(:,:,26);
save('Task3/m26.mat', 'mean_jean_walker');
save('Task3/cov26.mat', 'clarence_covarience_carlson');

T = table(N, Nerr, acc)
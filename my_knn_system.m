load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1637668/data.mat');
Xtrn = single(dataset.train.images) ./ 255.0;
Ctrn = single(dataset.train.labels);
Xtst = single(dataset.test.images) ./ 255.0;
Ks = [1;3;5;10;20];
tic
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, Ks);
toc

Ctrues = dataset.test.labels;

confusions = cell(5,1);
accs = zeros(5,1);
Nerrs = zeros(5,1);
N = [7800;7800;7800;7800;7800];

for k = 1:size(Ks)
    [CM, acc] = my_confusion(Ctrues, Cpreds(:,k));
    confusions{k,1} = CM;
    Nerrs(k,1) = sum(sum(CM)) - trace(CM);
    accs(k,1) = acc;
end

cm1 = confusions{1};
cm2 = confusions{2};
cm3 = confusions{3};
cm4 = confusions{4};
cm5 = confusions{5};

save('cm1.mat', 'cm1');
save('cm2.mat', 'cm2');
save('cm3.mat', 'cm3');
save('cm4.mat', 'cm4');
save('cm5.mat', 'cm5');

T = table(Ks, N, Nerrs, accs)

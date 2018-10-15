load('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/s1637668/data.mat');


%{
Xtrn = [1 0 0 0 1 1 1 1;
        0 0 1 0 1 1 0 0;
        0 1 0 1 0 1 1 0;
        1 0 0 1 0 1 0 1;
        1 0 0 0 1 0 1 1;
        0 0 1 1 0 0 1 1;
        
        0 1 1 0 0 0 1 0;
        1 1 0 1 0 0 1 1;
        0 1 1 0 0 1 0 0;
        0 0 0 0 0 0 0 0;
        0 0 1 0 1 0 1 0];
        %0 1 0 0 0 1 0 0];

Ctrn = [1;
        1;
        1;
        1;
        1;
        1;
        
        2;
        2;
        2;
        2;
        2];    
    
Xtst = [1 0 0 1 1 1 0 1;
        0 1 1 0 1 0 1 0];

my_bnb_classify(Xtrn, Ctrn, Xtst, 1)
    %}
Xtrn = dataset.train.images;
Ctrn = dataset.train.labels;
Xtst = dataset.test.images;
threshold = 1;

tic
[Cpreds] = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold);
toc
[c, acc] = my_confusion(dataset.test.labels, Cpreds);
        %}
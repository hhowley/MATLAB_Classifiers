Xtrn = [4 4; 3 4; 4 3; 0 0; 1 0; 0 1];
Ctrn = [1; 1; 1; 2; 2; 2];
Xtst = [1 1; 3 3];
Ks = [1;2;3;4];
Cpreds = my_knn_classify(Xtrn, Ctrn, Xtst, Ks)
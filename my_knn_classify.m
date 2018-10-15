function [Cpreds] = my_knn_classify(Xtrn,Ctrn,Xtst,Ks)
    X = Xtst;
    Y = Xtrn;

    [N,D] = size(X);
    [M,D] = size(Y);
    [L,o] = size(Ks);

    Cpreds = zeros(N,L);

    XX = sum(X .* X,2);
    YY = sum(Y .* Y,2);

    XXM = repmat(XX,1,M);
    YYM = repmat(YY,1,N);

    DI = XXM - 2 * X * Y' + YYM';

    [~, sortedindexes] = sort(DI, 2, 'ascend');

    %sortedindexes = sortedindexes';

    for k = 1:L
        cutindexes = sortedindexes(:,1:Ks(k));
        pre = mode(Ctrn(cutindexes),2);
        Cpreds(:,k) = pre;
    end

end

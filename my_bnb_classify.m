function [ Cpreds ] = my_bnb_classify(Xtrn, Ctrn, Xtst, threshold)
    Xtrn = single(Xtrn);
    Xtst = single(Xtst);
    
    bxtrn = Xtrn >= threshold;
    bxtst = Xtst >= threshold;
    
    uclasses = unique(Ctrn);
    prosteriors = zeros(size(uclasses,1),size(Xtrn,2));
    
    for p = 1:size(uclasses,1)
        cindexes = find(Ctrn == uclasses(p));
        prosteriors(p,:) = sum(bxtrn(cindexes,:),1)./size(cindexes,1);
    end
    
    prosteriors(prosteriors == 0) = 1.0E-20;
    
    classprobs = bxtst*log(prosteriors') + bsxfun(@minus, 1, bxtst)*log(bsxfun(@minus, 1, prosteriors)');
    
    [~,Cpreds] = max(classprobs,[],2);
end


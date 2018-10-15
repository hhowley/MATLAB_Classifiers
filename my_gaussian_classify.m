function [Cpreds, Ms, Covs] = my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon)

    Cpreds = zeros(size(Xtst,1),1);
    
    uclasses = unique(Ctrn);
    
    Ms = zeros(size(Xtrn,2),size(uclasses,1));
    Covs = zeros(size(Xtst,2),size(Xtst,2),size(uclasses,1));
    loglike = zeros(size(Xtst,1),size(uclasses,1));
    
    for c = 1:size(unique(Ctrn),1)
        CXtrn = Xtrn((Ctrn==c),:);
        Ms(:,c) = (sum(CXtrn,1)/size(CXtrn,1))';
        Mtrn = CXtrn - repmat(Ms(:,c)',size(CXtrn,1),1);
        Covs(:,:,c) = (Mtrn'*Mtrn)/size(CXtrn,1) + (eye(size(Xtrn,2))*epsilon);
        Mtst = Xtst - repmat(Ms(:,c)',size(Xtst,1),1);
        loglike(:,c) = (-0.5)*sum(Mtst*inv(Covs(:,:,c))'.*Mtst,2)-0.5*(logdet(Covs(:,:,c)));
    end
    
    [~,Cpreds] = max(loglike,[],2);
end


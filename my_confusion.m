function [CM, acc] = my_confusion(Ctrues, Cpreds)
    
    [A, B] = size(Ctrues);
    CM = zeros(A,A);
    
    for k = 1:A
        ktrue = Ctrues(k,1);
        kpred = Cpreds(k,1);
        CM(ktrue,kpred) = CM(ktrue,kpred) + 1;
    end
    
    CMtrace = trace(CM);
    CMsum = sum(sum(CM));
    acc = CMtrace / CMsum;
    
end



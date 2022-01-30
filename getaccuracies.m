function [ oAcc, aAcc, order, classAccs, C, precision, recall ] = getaccuracies( labels, gt )

[ C, order ] = confusionmat( labels, gt );
k = 0;
for j = 1:size(C,1)
    k = k + C(j,j);
end
n = sum(C(:));
oAcc = k/n * 100;

classAccs = zeros(length(order),1);
aAcc = 0;
for j = 1:size(C,1)
    k = C(j,j);
    %n = sum( C(:,j) ) + sum( C(j,:) ) - C(j,j);
    n = sum(C(:,j));
    classAccs(j) = k/n;
    aAcc = aAcc + k/n;
    %aAcc = aAcc + sum(C(j,j)) / sum(C(:,j)) * 100;
end
aAcc = aAcc / size(C,1) * 100;
classAccs = classAccs * 100;
%aAcc = aAcc / length(C);
precision=diag(C)./sum(C,2);
recall=diag(C)./sum(C,1)';
end


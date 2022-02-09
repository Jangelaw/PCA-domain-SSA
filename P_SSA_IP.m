% clear
addpath(genpath('libSVM'));
load('data\Indian_pines_corrected.mat');
load('data\Indian_pines_gt.mat');
image=double(indian_pines_corrected); %% with noise
GT=indian_pines_gt;

index = unique(GT);
index(index==0) = [];
class_num=length(index);


[a,b,c]=size(image);
I=reshape(image,[a*b,c]);
iter=10;

for i = 1:class_num
    GT(GT == index(i)) = i;
    num_of_label(i) = sum(GT(:)==i);
end

clear i

nPc=90; %% IP-90, PU-40, SA- 20
para=[10,10];

L=10;
u=L; v=L; first=1; last=1;
trainsample_num=ones(class_num,1)*[5,10,15,20,25,30];
trainsample_num(7,:)=13; %% liangpei zhang, neuralcomputing 2019
trainsample_num(9,:)=10; %% liangpei zhang, neuralcomputing 2019
trainsample_num=[floor(num_of_label.*0.05)',floor(num_of_label.*0.1)',trainsample_num];

t1 = clock;
Band_Project2=PCA_INSIDE(100,I);
%% FPCA'
comp=para(1,1);
H=para(1,2);
Band_Project1=FPCA(comp,H,I')';
I1=reshape(Band_Project1,[a,b,comp]);

input_feature=zeros(a,b,nPc(1,1)+comp);
%% PCA

I2=reshape(Band_Project2(:,1:nPc(1,1)),[a,b,nPc(1,1)]);
input_image=cat(3,I1,I2);
for i=1:nPc(1,1)+comp    %1:band
[input_feature(:,:,i)]=SSA_2Ds(input_image(:,:,i),u,v,first,last);
end
clear i
t2 = clock;
for k=1:size(trainsample_num,2)
%% training
    oAcc=zeros(iter,1);
    aAcc=zeros(iter,1);
    kp=zeros(iter,1);
    C=cell(iter,1);
    for i=1:iter
        [oAcc(i),aAcc(i),cAcc(i,:),kp(i),C{i},R{i}]=training(input_feature,GT,num_of_label,trainsample_num(:,k)',class_num);
    end
    OA(k)=mean(oAcc);
    OA_std(k)=std(oAcc);
    AA(k)=mean(aAcc);
    AA_std(k)=std(aAcc);
    KP(k)=mean(kp*100);
    KP_std(k)=std(kp*100);
    CA(k,:)=mean(cAcc,1);
    CA_std(k,:)=std(cAcc,1);
    C2{k}=sum(cat(3,C{:}),3)/iter;
    Rec{k}=mean(cat(3,R{:}),3);
    clear image2
end
t3=clock;
T=etime(t2,t1)+etime(t3,t2)/(size(trainsample_num,2)*iter); %% time
save(['IP_result.mat'],'OA','AA','KP','CA','OA_std','AA_std','KP_std','CA_std','C2','Rec');
function [oAcc,aAcc,classAccs,kp,C2,Rec]=training(image2,GT,num_of_label,trainsample_num,class_num)
[a,b,nPc]=size(image2);

clear i

train_labels=double(getlabeled_dynamic2(GT, trainsample_num,class_num));
test_labels = GT - train_labels;
num_of_test=num_of_label-trainsample_num;
%% classification
index_train=find(train_labels>0);
index_test=find(test_labels>0);
I2=reshape(image2,[a*b,nPc]);
GT2=reshape(GT,[a*b,1]);
label_tra=GT2(index_train,:);
label_tes=GT2(index_test,:);
data_tra = I2(index_train,:);
data_tes = I2(index_test,:);

bestc=1024;bestg=0.125;
[Mnorm_tra,Mnorm_tes,ps] = scaleForSVM(data_tra,data_tes,0,1);
% [bestacc,bestc,bestg] = SVMcgForClass(label_tra,Mnorm_tra,-10,10,-10,10,10,1,1,4.5);

cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model = svmtrain(label_tra, Mnorm_tra,cmd);
[ptrain_label, train_accuracy] = svmpredict(label_tra, Mnorm_tra, model);
[ptest_label, test_accuracy] = svmpredict(label_tes, Mnorm_tes, model);
Rec=zeros(a,b);
Rec(index_train)=ptrain_label;
Rec(index_test)=ptest_label;
[oAcc, aAcc, ~, classAccs,C] = getaccuracies(ptest_label, label_tes);
[~,kp]=kappa(C);
C2=C./repmat(num_of_test,[length(num_of_label),1]);
end

function output=PCA_INSIDE(comp,data)
[N,~] = size(data);
u = mean(data);
data = data - repmat(u,N,1);

[coeff score latent] = pca(data);
V = coeff(:, 1:comp);

output = data * V;
end

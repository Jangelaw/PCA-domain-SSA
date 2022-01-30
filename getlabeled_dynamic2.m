function [labeled, unlabeled,train_index,test_index]= getlabeled_dynamic2(groundtruth, num,class_num)
%%%=============================================
%[labeled, unlabeled]= getlabeled(groundtruth, precent) gets labeled
%samples randomly from the groundtruth to train the projection matrix W
%      output:
%              groundtruth   ----the ground truth of the hyperspectral
%                                image
%              precent       ----the percentage of labeled samples used to
%                                train SVM classifier
%      input:  
%              labeled       ----the mask used to train SVM classifier
%              unlabeled     ----the test samples
%
train_index=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B= groundtruth;
for i=1:class_num
    tempindex=find(groundtruth==i);
    xx=randperm(length(tempindex));
   train_index=[train_index,tempindex(xx(1:num(i)))'];
end
groundtruth(train_index)=0;

labeled=B-groundtruth;
unlabeled=groundtruth;

test_index=find(unlabeled~=0);



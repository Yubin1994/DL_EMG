clear
clc


ftrDir = 'G:\ÎâÀÖ\baseline\baseline\·½°¸Ò»\Î´¹éÒ»»¯\';
saveDir = 'G:\ÎâÀÖ\10_10_ULDA_LDC\';

lenCls = 7;
train_d = load('G:\ÎâÀÖ\°üÂç18_03_07\ÂË²¨\baseline_50ms_10x10_Norm\7cls\data_train.mat');
train_d = train_d.data_train;
train_d = reshape(train_d,size(train_d,1),100);
train_label = load('G:\ÎâÀÖ\°üÂç18_03_07\ÂË²¨\baseline_50ms_10x10_Norm\7cls\label_train.mat');
train_label = train_label.label_train + 1;
test_d = load('G:\ÎâÀÖ\°üÂç18_03_07\ÂË²¨\leftTop_50ms_Norm\7cls\data_test.mat');
test_d = test_d.data_test;
test_d = reshape(test_d,size(test_d,1),100);
test_label = load('G:\ÎâÀÖ\°üÂç18_03_07\ÂË²¨\leftTop_50ms_Norm\7cls\label_test.mat');
test_label = test_label.label_test +1;


transFtr = A1_4ULDA_EMG(train_d,train_label);
ftrTrans = train_d*transFtr;

for idcls = 1:lenCls
    staend = find(train_label==idcls);
    ftrArr = ftrTrans(staend,:);
    Train_Class(1,idcls).transFtr = transFtr;
    
    Train_Class(1,idcls).LDC.Mean = mean(ftrArr);
    Train_Class(1,idcls).LDC.CovM = cov(ftrArr);
    Train_Class(1,idcls).LDC.CovM = diag(diag(Train_Class(1,idcls).LDC.CovM));
    Train_Class(1,idcls).LDC.ICov = pinv(Train_Class(1,idcls).LDC.CovM);
    Train_Class(1,idcls).LDC.DCov = det(Train_Class(1,idcls).LDC.CovM);
end

[sam,~] = size(test_d);
clsSam = zeros(lenCls,1);
cor = zeros(lenCls,lenCls);
for k = 1:sam
    
    clsDis = [];
    arr = test_d(k,:);
    for idcls = 1:lenCls
        ftrarr = arr*Train_Class(1,idcls).transFtr;
        Loglik = -0.5*(ftrarr-Train_Class(1,idcls).LDC.Mean)...
                *Train_Class(1,idcls).LDC.ICov*(ftrarr-Train_Class(1,idcls).LDC.Mean)'...
                -0.5*log(Train_Class(1,idcls).LDC.DCov);
        clsDis = [clsDis,Loglik];
    end
    index = find(clsDis==max(clsDis));
    cor(index,test_label(k)) = cor(index,test_label(k))+1;
    clsSam(test_label(k),1) = clsSam(test_label(k),1)+1;
end
for i = 1:lenCls
    for j = 1:lenCls
        cor(i,j) = cor(i,j)/clsSam(j,1);
    end
end
predict = 100*sum(cor)/sam;
save([saveDir,'predict.mat'],'predict');
save([saveDir,'cor.mat'],'cor');
    
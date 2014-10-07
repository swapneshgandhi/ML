%scan microsoft letor data set

clc
clear all
close all
init_fid=fopen('C:\Users\Swapnesh\Documents\books_fall-13\ML\project1\dataset.txt');
y = 1;
i=1;
tline = fgetl(init_fid);
while ischar(tline)

    tline=textscan(tline,'%s','delimiter',[' ',':'],'commentstyle','#');
    c=1;
    for i=5:length(tline{1,1})
        target(y,1)=str2double(tline{1,1}{1,1});
        init1{y,i-4}=tline{1,1}{i,1};
    end
    for i=1:2: length(tline{1,1})-4
        final(y,c)=str2double(init1{y,i});
        c=c+1;
        
    end
    
    tline = fgetl(init_fid);
    y=y+1;
end
fclose(init_fid);
train_d=final(1:6085,:);
validation=final(6086:7605,:);
test=final(7606:15211,:);
train_target=target(1:6085,:);
validation_target=target(6086:7605,:);
test_target=target(7606:15211,:);

mean_f=mean(train_d);
var_t=(var(train_d,0,1)).^2;

save('project1_data.mat','final','train_d','validation','test','target','train_target','test_target','validation_target','mean_f','var_t');

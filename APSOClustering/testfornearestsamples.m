clear
close all
clc

dataset=randi([1,100],10,10);

numofsamples=10;

centers=rand(3,10);

numofrepres=6;

temprepsamp=randi([1,numofsamples],1,numofrepres);

represamples=ChoosingUniqueRepSamp(temprepsamp,size(dataset,1));

% represtsamp=ChoosingUniquesamples(temprepsamp,numofsamples

numofclusters=3;

distance=dist(centers,dataset(represamples,:)');

for i=1:numofclusters
    
    p=distance(i,:);
    
    [~,ind]=min(p)
    
    assignedsamp=dataset(represamples(ind),:);
    
    centers(i,:)=assignedsamp;
    
    p=[];
    
    assignedsamp=[];
  
end
    

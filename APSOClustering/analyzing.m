clear 
close all
clc

% C=load('clustercenters.mat');

numofselectedfeatures=32;

dataset=xlsread('dim032.xlsx');

[~,numoffeatures]=size(dataset);
        for  i=1:numoffeatures


            data=dataset(:,i);

            div(i)=std(data);

        end


 div=div';
        
 [diversity featurenumber]=sort(div,'descend');
        
 [selectedfeatures newdataset]=RemovingFeatures(dataset,numofselectedfeatures);

realclusternumber=16;

% centroids=C.centroids;

centroids=xlsread('centroids3.xlsx');

labels=[1:size(centroids,1)]';

model=fitcknn(centroids,labels,'NumNeighbors',1);

Dataclustering=model.predict(newdataset);

Reallabels=xlsread('dim032labels.xlsx');

numofclusters=size(centroids,1);

distance=zeros(numofclusters);

% D=load('samplespercluster.mat');
% 
% labels=D.Dataclustering;

for i=1: numofclusters
    
    numofsamples(i)=numel(find(Dataclustering==i));
    
end


for j=1:realclusternumber
    
    realnumofSam(j)=numel(find(Reallabels==j));
    
end

for i=1:numofclusters
    
    for j=1:numofclusters
        
        distance(i,j)=dist(centroids(i,:),centroids(j,:)');
        
    end
    
    
end



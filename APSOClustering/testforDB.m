clear
close all
clc


dataset=xlsread('dim032.xlsx');

numofselectedfeatures=32;
% Realcenters=xlsread('Realcenters.xlsx');

[~,numoffeatures]=size(dataset);
        for  i=1:numoffeatures

            data=dataset(:,i);

            div(i)=std(data);

        end


 div=div';
        
 [diversity featurenumber]=sort(div,'descend');
        
 [selectedfeatures newdataset]=RemovingFeatures(dataset,numofselectedfeatures);

% C=load('clustercenters.mat');

realclusternumber=16;

% centroids=C.centroids;

centroids=xlsread('centroids1.xlsx');

Reallabels=xlsread('dim032labels.xlsx');

numofclusters=size(centroids,1);

labels=[1:size(centroids,1)]';

model=fitcknn(centroids,labels,'NumNeighbors',1);

Dataclustering=model.predict(newdataset);

DB=evalclusters(newdataset,Dataclustering,'DaviesBouldin') 

%SIL=evalclusters(newdataset,Dataclustering,'Silhouette')
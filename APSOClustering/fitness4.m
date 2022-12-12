function [ f ] = fitness4( numofselectedfeatures,numofcentroids,dataset,centroids )
% here i used the toolbox for calculating silhouette


labels=[1:size(centroids,1)]';

model=fitcknn(centroids,labels,'NumNeighbors',1);
p=model.predict(dataset);

f=evalclusters(dataset,p,'Silhouette');

end


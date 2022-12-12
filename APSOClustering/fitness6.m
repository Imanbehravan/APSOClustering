function [ f ] = fitness6( numofselectedfeatures,numofcentroids,dataset,centroids )

% evaluating the solutions by Davies-Bouldin index

labels=[1:size(centroids,1)]';

model=fitcknn(centroids,labels,'NumNeighbors',1);
p=model.predict(dataset);

f=evalclusters(dataset,p,'CalinskiHarabasz');

end


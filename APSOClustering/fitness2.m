function [ F ] = fitness2(dataset,centroids,K )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

labels=[1:size(centroids,1)]';

model=fitcknn(centroids,labels,'NumNeighbors',1);

IDX=model.predict(dataset);

F=1-silhcoef(dataset,IDX,K);

end


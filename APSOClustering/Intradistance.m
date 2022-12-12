function [ Distance ] = Intradistance( dataset,centroids,Dataclustering,numofclusters )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


for i=1:numofclusters
    
    index=find(Dataclustering==i);
    samples=dataset(index,:);
    
    if numel(samples) > 0
        
%              intradist(i)=sum(dist(samples,centroids(i,:)'))/size(samples,1);
                    intradist(i)=sum(dist(samples,centroids(i,:)'));
    
    end
    
    if numel(samples) == 0
        
        intradist(i)=0;
        
    end 
    
    
end

% n=numel(find(intradist>0));

% Distance=sum(intradist)/numel(intradist);
% Distance=(sum(intradist))/n;
       Distance=sum(intradist);
end


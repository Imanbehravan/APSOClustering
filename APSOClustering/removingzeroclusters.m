function [ newpop, newbestpos, numofclusters ] = removingzeroclusters( pop,bestposition,numofselectedfeatures,newdataset )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here




% numofselectedfeatures=2;

bestcentroids=reshape(bestposition,numofselectedfeatures,[])';

% dataset=xlsread('s1.xlsx');
% 
% Reallabels=xlsread('s1labels.xlsx');
% 
% % Realcenters=xlsread('Realcenters.xlsx');
% 
% [~,numoffeatures]=size(dataset);
%         for  i=1:numoffeatures
% 
% 
%             data=dataset(:,i);
% 
%             div(i)=std(data);
% 
%         end
% 
% 
%  div=div';
%         
%  [diversity featurenumber]=sort(div,'descend');
%         
%  [selectedfeatures newdataset]=RemovingFeatures(dataset,numofselectedfeatures);
 
labels=[1:size(bestcentroids,1)]';

model=fitcknn(bestcentroids,labels,'NumNeighbors',1);

p=model.predict(newdataset);

for i=1:size(bestcentroids,1)
    
    samplesperclusters(i)=numel(find(p==i));
    
end

nonzeroclusters=find(samplesperclusters>0);

index=1;

for j=1:numel(nonzeroclusters)
    
    bestsolution(1,index:index+numofselectedfeatures-1)=bestposition(1,(numofselectedfeatures*(nonzeroclusters(j)-1)+1):numofselectedfeatures*nonzeroclusters(j));
    
    index=index+numofselectedfeatures;
    
end

centroidsolution=reshape(bestsolution,numofselectedfeatures,[])';

newpop=zeros(size(pop,1),numel(nonzeroclusters));

index=1;

for h=1:numel(nonzeroclusters)
    
    newpop(:,index:index+numofselectedfeatures-1)=pop(:,(numofselectedfeatures*(nonzeroclusters(h)-1)+1):numofselectedfeatures*nonzeroclusters(h));
    
    index=index+numofselectedfeatures;
    
end

newbestpos=bestsolution;



numofclusters=numel(nonzeroclusters);

end


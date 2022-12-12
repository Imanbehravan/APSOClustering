function  F = fitness3(numofselectedfeatures,numofcentroids,dataset,centroids)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%        numofselectedfeatures=x(2);
       
%        numofcentroids=x(1);
       
%        centroids=reshape(x,numofselectedfeatures,[])';
       
%       selectedfeatures=sort(featurenumber(1:numofselectedfeatures));

      labels=[1:size(centroids,1)]';

%         for i=1:numel(selectedfeatures)
%     
%             data2(:,i)=dataset(:,selectedfeatures(i));
% 
%         end
        
%         newdataset=data2;
        
        model=fitcknn(centroids,labels,'NumNeighbors',1);
        p=model.predict(dataset);
        
%         F=Dbouldin(dataset,numofcentroids,Dataclustering,centroids);

%         a1=Intradistance(dataset,centroids,Dataclustering,numofcentroids);

        a1=Intradistance(dataset,centroids,p,numofcentroids);
        
%         a2=ClusterDistance(centroids,numofcentroids);
       
        
%         F=1-silhcoef(dataset,Dataclustering,numofcentroids);
%             F=Coeff1*a1+Coeff2*(1/a2);
%                 F=Coeff1*a1;
        F=a1;

end


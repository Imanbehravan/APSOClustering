function [ assignedcenters ] = AssingningNearestSamp( centroids,dataset,numofrepres,numofclusters )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here


assignedcenters=zeros(size(centroids));

numofsamples=size(dataset,1);

centers=centroids;

% numofrepres=6;

temprepsamp=randi([1,numofsamples],1,numofrepres);

represamples=ChoosingUniqueRepSamp(temprepsamp,size(dataset,1));

% represtsamp=ChoosingUniquesamples(temprepsamp,numofsamples

% numofclusters=3;

distance=dist(centers,dataset(represamples,:)');

assignedsamp=[];

for i=1:numofclusters
    
    p=distance(i,:);
    
    [~,ind]=min(p);
    
    if numel(find(assignedsamp==ind))>0
        
%         ind=randi([1,numofsamples]);

        check=0;
        
        while (check==0)
            
            ind=randi([1,numofrepres]);
            
            if (numel(find(assignedsamp==ind))==0)
                
                check=1;
                
            end
            

        end
        
    end
    
    assignedsamp(i)=ind;
    
    chosensample(i)=represamples(ind);
    
%     assignedsamp=dataset(represamples(ind),:);
    
%     assignedcenters(i,:)=assignedsamp;
    
    p=[];
    
%     assignedsamp=[];
  
end
    
assignedcenters=dataset(chosensample,:);


end


function [ newposition newcentroids fitness ] = revise( cost,position,nvar,numofselectedfeatures,numofclusters,newdataset )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
disp(' ')
disp('------------revise-----------');
disp('---------processing---------- ');

disp(position);

check=0;

if  isnan(cost)
    
    check = 1;
    
end

while (check==1)
    
    position = position + randi([-10 10],1,nvar);
    
    newposition = position;
    
    newcentroids=reshape(newposition,numofselectedfeatures,[])';
    
    fitness=fitness3(numofselectedfeatures,numofclusters,newdataset,newcentroids);
    
    if isnan(fitness)
        
        check=1;
        
    end
    
    if ~isnan(fitness)
        
        check=0;
        
    end
    
   
end

disp('------------end of revise-----------');
disp(' ');

end


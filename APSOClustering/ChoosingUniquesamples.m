function [ F ] = ChoosingUniquesamples( chosensample,numofsamples,numofclusters )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


check=0;

while(check==0)
 
    for i=1:numel(chosensample)

        a=chosensample(i);

        Rep=find(chosensample==a);

        if (numel(Rep)>1)

            for j=1:numel(Rep)-1

                chosensample(Rep(j))=randi([1,numofsamples]);

            end

        end


    end
    
    f=unique(chosensample);
    
    if (numel(f)==numofclusters)
        
        check=1;
        
    end

end

F=chosensample;

end


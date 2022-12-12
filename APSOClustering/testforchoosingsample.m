clear
close all
clc

chosensample=[1 16 18 19 12 13 15 20 26 25];

numofsamples=30;

numofclusters=10;

check=0;

while(check==0)
 
    for i=1:numel(chosensample)

        a=chosensample(i);

        Rep=find(chosensample==a);

        if (Rep>1)

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

chosensample
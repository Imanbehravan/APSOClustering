function [ F ] = ChoosingUniqueRepSamp( repressamples,sizeofdatast )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

numofrepsamp=numel(repressamples);


check=0;

while(check==0)
 
    for i=1:numel(repressamples)

        a=repressamples(i);

        Rep=find(repressamples==a);

        if (numel(Rep)>1)

            for j=1:numel(Rep)-1

                repressamples(Rep(j))=randi([1,sizeofdatast]);

            end

        end


    end
    
    f=unique(repressamples);
    
    if (numel(f)==numofrepsamp)
        
        check=1;
        
    end

end

F=repressamples;


end


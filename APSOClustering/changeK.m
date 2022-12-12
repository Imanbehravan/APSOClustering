function [ population,k ] = changeK(K,pop,AVG,m,numofselectedfeatures,xmin,xmax )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

r2=unifrnd(0,1);

if (K==2)
    
    r2=0.3;
    
end

    if (r2>=0)&&(r2<0.5)
        
        increase=round(unifrnd(1,10));
        
        newk=K+increase;
        
        S=zeros(m,increase*numofselectedfeatures);
        
        for i=1:m
            
              S(i,:)=rand*repmat(AVG,1,increase)+rand(1,increase*numofselectedfeatures)+randi([round(xmin) round(xmax)],1,increase*numofselectedfeatures);  
            
        end
        
        newpop=[pop S];
        
        population=newpop;
        
        k=newk;
          
    end
    
    if (r2>=0.5)&&(r2<=1)
        
        decrease=round(unifrnd(1,K-2));
        
        newk=K-decrease;
        
        newpop=pop(:,1:newk*numofselectedfeatures);
        
        population=newpop;
        
        k=newk;
          
    end


end


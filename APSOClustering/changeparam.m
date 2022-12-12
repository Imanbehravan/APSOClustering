function [ c1,c2,m,population,k ] = changeparam( C1,C2,M,pop,AVG,numofclusters,numofselectedfeatures,xmin,xmax )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

population=pop;

K=numofclusters;

k=K;

c1=C1;

c2=C2;

m=M;

r1=unifrnd(0,4);

r2=unifrnd(0,1);

if (r1>=0)&&(r1<1)

    if (r2>=0)&&(r2<0.5)
        
        c1=C1+unifrnd(0.1,1);
        
    end
    
    if (r2>=0.5)&&(r2<=1)
        
        c1=C1-unifrnd(0.1,1);
          
    end
    
    
end

if (r1>=1)&&(r1<2)

    if (r2>=0)&&(r2<0.5)
        
        c2=C2+unifrnd(0.1,1);
        
    end
    
    if (r2>=0.5)&&(r2<=1)
        
        c2=C2-unifrnd(0.1,1);
          
    end
    
    
end


if (r1>=2)&&(r1<3)

    if (r2>=0)&&(r2<0.5)
        
        increase=round(unifrnd(1,10));
        
        m=M+increase;
        
        for i=1:increase
            
           population(M+i,:)=rand*repmat(AVG,1,numofclusters)+rand(1,numofclusters*numofselectedfeatures)+randi([round(xmin) round(xmax)],1,numofclusters*numofselectedfeatures); 
            
        end
        
        
    end
    
    if (r2>=0.5)&&(r2<=1)
        
        m=M-round(unifrnd(1,0.3*M));
        
        population=population(1:m,:);
          
    end
    
    
end


if (r1>=3)&&(r1<4)

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


end


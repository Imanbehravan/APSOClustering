function  sil=silhcoef(dataset,IDX,k)


IDX=IDX';
index3=0;
s=silhouette(dataset,IDX);

for i=1:k
    
    index=find(IDX==i);
    
    index2=s(index);
    n=numel(index2);
    
    if numel(index)~=0
          S(i)=sum((index2))/n;
    end
    
    if  numel(index)==0
        S(i)=0;
        index3=index3+1;
    end
    
    
    index=[];
    index2=[];
    n=[];
end

k=k-index3;


sil=(1/k)*sum(S);


end


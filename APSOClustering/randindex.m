function [ RI ] = randindex( labels,solution )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here




% dataset=xlsread('MyData3');
% labels=xlsread('MyData3labels');

samecluster=0;
differentcluster=0;

samelabel=0;
differentlabel=0;
a=0;
b=0;
c=0;
d=0;

% S=[3	3	3	3	3	1	3	1	3	3	3	1	3	1	1	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	1	3	3	3	3	3	3	3	3	3	3	3	3	2	3	3	3	3	3	3	3	3	3	3	3	1	1	2	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	1	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	3	2	3	3	3	1	3	3	3	3	2	3	3	3	3	3	3	3	3	3	3	2	3	3	3	3	3	3	3	2	3	3	3	3	2	3	3	3	1	3	3	3	3	3	3	3	3	3	3	3	3	1	3	3	3	3	3	3	3	3	3	3	3	3	2	3	3	3	3	3	3	3	3	3	3	3	3	3	3	2	3	1	3	3	3	3	3	3	1	3	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	3	1	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	1	1	3	1	2	2	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	3	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	3	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	1	3	1	2	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	3	1	1	1	1	1	1	1	1	1	1	2	1	2	1	2	1	1	1	1	2	1	1	1	1	1	1	1	1	1	1	1	3	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	2	1	1	1	1	1	1	1	1	1	1	2	3	1	1	1	1	1	1	1	1	1	1	1	1	1	3	1	1	1	1	1	3	1	1	1	3	1	1	1	3	1	1	3	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	3	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	2	3	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	1	2	2	2	2	2	1	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	1	2	2	2	2	2	2	2	2	2	2	2	3	2	2	2	2	2	2	2	2	2	2	2	2	2	2	3	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	1	2	3	2	2	2	2	2	2	2	2	2	2	2	3	3	2	2	2	2	2	2	3	2	2	2	2	2	2	2	2	2	2	2	2	3	2	2	3	2	3	2	2	2	2	2	2	2	2	2	2	1	2	2	2	2	2	1	2	2	2	2	2	2	2	2	2	3	2	2	2	2	2	2	2	2	3	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2];

% solution=S(1,2:end);
% 
% ind1=find(solution==1);
% x1=dataset(ind1,:);
% 
% ind2=find(solution==2);
% x2=dataset(ind2,:);
% 
% ind3=find(solution==3);
% x3=dataset(ind3,:);

% ind4=find(solution==4);
% x4=dataset(ind4,:);

% plot(x1(:,1) ,x1(:,2), 'bo');
% hold on
% plot(x2(:,1) ,x2(:,2), 'sr');
% hold on
% plot(x3(:,1) ,x3(:,2), 'kp');
% % hold on
% plot(x4(:,1) ,x4(:,2), 'gp');
% hold on

for i=1:numel(solution)
    
    firstobjectcluster=solution(i);
    firstobjectlabel=labels(i);
    
 for j=i+1:numel(solution)
        
        secondobjectcluster=solution(j);
        secondobjectlabel=labels(j);
        
        if firstobjectcluster==secondobjectcluster
            
            samecluster=1;
            
        end
        
        if firstobjectcluster~=secondobjectcluster
            
            differentcluster=1;
            
        end
        
        if firstobjectlabel==secondobjectlabel
            
            samelabel=1;
            
        end
        
        if firstobjectlabel~=secondobjectlabel
            
            differentlabel=1;
            
        end
        
        if (samecluster==1)&&(samelabel==1)
            
            a=a+1;
            
        end
        
        if (differentcluster==1)&&(differentlabel==1)
            
            b=b+1;
            
        end
        
        if (samecluster==1)&&(differentlabel==1)
            
            c=c+1;
            
        end
        
        if (differentcluster==1)&&(samelabel==1)
            
            d=d+1;
            
        end
        
        samecluster=0;
        differentcluster=0;
        samelabel=0;
        differentlabel=0;
        
       
    end
    
    
    
    
end

clusteringvalidation=(a+b)/(a+b+c+d);


RI=clusteringvalidation;



end


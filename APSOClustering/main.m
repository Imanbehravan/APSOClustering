clc;
clear;
close all

%% if your dataset is labeled and you know the number of clusters then import it here

realclusternumber=2;

%% number of features of the dataset

numofselectedfeatures=32;

%% APSO parameters
lengthoftree=3;

numofPSOs=10;

FuncIter=150;

MainPSOiter=600;

%% importing dataset and the real labels (if exist)

dataset=xlsread('g2-32-80.xlsx');

Reallabels=xlsread('g2-32-80labels.xlsx');

%% 
numofrepres=round(0.1*size(dataset,1));

[~,numoffeatures]=size(dataset);

        for  i=1:numoffeatures


            data=dataset(:,i);

            div(i)=std(data);

        end


 div=div';
        
 [diversity featurenumber]=sort(div,'descend');
        
 [selectedfeatures newdataset]=RemovingFeatures(dataset,numofselectedfeatures);
        
 AVG=mean(newdataset);
 
xmin=min(min(newdataset));

xmax=max(max(newdataset));
  
%% defining parameters 

param.numofclusters=[];

param.inspectedK=[];

param.m=[]; % number of beginning population for each tree

param.pop=[];

%% initializing each tree 

S=size(newdataset);

%% 
tic;

for i=1:numofPSOs
    
    disp(' ');

    disp(['----------------------tree number: ' num2str(i) '--------------------']);

    disp(' ');
    
    index=1;

    bestresult=[];
    
    result=[];
    
    result.position=[];
    
    result.cost=[];
    
    result.centroids=[];
    
    param.numofclusters=[];
    
    param.inspectedK=[];

    param.m=[]; % number of beginning population for each tree
 
    param.pop=[];
    
    param.cost=[];
    
    param.bestpos=[];
    
    param.numofclusters=round(unifrnd(3,sqrt(S(1))));
  
    param.m=5;

    nvar=param.numofclusters*numofselectedfeatures;

    population=zeros(param.m,nvar);

    for j=1:param.m

        p=randi([1,size(newdataset,1)],1,param.numofclusters);
        
        chosensamples=ChoosingUniquesamples(p,size(newdataset,1),param.numofclusters);
        
        index=1;
        
        for h=1:param.numofclusters
                   
          population(j,index:index+numofselectedfeatures-1)=newdataset(chosensamples(h),:);
            
          index=index+numofselectedfeatures;
        
        end
        
    end

    param.pop=population;
    
    disp(' ');

    disp(['-------------PSO number: ' num2str(1) '--------------']);

    disp(' ');
    
    R=PSOfunc(numofselectedfeatures,param.numofclusters,newdataset,FuncIter,param.m,param.pop,numofrepres,xmin,xmax);
    
    result=R;
    
    param.inspectedK=R.inspectedK;
            
    PSOleaf(1).m=param.m;
            
    PSOleaf(1).pop=param.pop;
            
    PSOleaf(1).k=param.numofclusters;
     
    PSOleaf(1).inspectedK=result.inspectedK;
            
    PSOleaf(1).silind=result(1).cost;

    param.cost=result(1).cost;
     
    param.bestpos=result(1).position;
    
    bestresult=result(1);
    
        for l=2:lengthoftree
            
            [newpop,newk]=changeK2(param.numofclusters,param.pop,AVG,param.m,numofselectedfeatures,xmin,xmax);
            
            disp(' ');

            disp(['-------------PSO number: ' num2str(l) '--------------']);

            disp(' ');
            
            result(l)= PSOfunc(numofselectedfeatures,newk,newdataset,FuncIter,param.m,newpop,numofrepres,xmin,xmax);
            
            PSOleaf(l).m=param.m;
            
            PSOleaf(l).pop=newpop;
            
            PSOleaf(l).k=newk;
            
            PSOleaf(l).inspectedK=result(l).inspectedK;
            
            PSOleaf(l).silind=result(l).cost;

            if result(l).cost<bestresult.cost

                param.numofclusters=newk;

                param.pop=newpop;
                
                param.cost=result(l).cost;
                
                param.inspectedK=result(l).inspectedK;
                
                param.bestpos=result(l).position;

                bestresult=result(l);

            end

            newpop=[];
            newk=[];
            
        end
        
        finalresult(i)=bestresult;
        
        finalparam(i)=param;
        
        figure;
        
        plot([PSOleaf.silind],'--o','markersize',5,'markerfacecolor','r','markeredgecolor','r','linewidth',1.5);
        
        name=['treePSO ' , num2str(i)];
        
        title(name);
        
        ylabel('Silhouette');
        
        xlabel('leaf');
        
        save(name,'PSOleaf');
        
        PSOleaf=[];

end

d=[finalresult.cost];

[~,index]=min(d);

selectedparam=finalparam(index);

[newpop, newbestpos, k]=removingzeroclusters(selectedparam.pop,selectedparam.bestpos,numofselectedfeatures,newdataset);

disp(' ');

disp('--------------------Main PSO-------------------');

 mainPSOout=PSOfunc2(numofselectedfeatures,k,newdataset,MainPSOiter,selectedparam.m,newpop,numofrepres,xmin,xmax);

toc;

save('result.mat','mainPSOout');

Centroids=mainPSOout.centroids;

figure

plot(newdataset(:,1),newdataset(:,2),'o');

hold on 

for j=1:k
    
    plot(Centroids(j,1),Centroids(j,2),'k*','markersize',10,'linewidth',1.5);
    
    hold on
    
end

%% calculating rand index

disp(' ');
disp('--------calculating the rand index-------');
disp(' ');

g=load('result.mat');
     
centroids=g.mainPSOout.centroids;
  
labels=[1:size(centroids,1)]';
  
model=fitcknn(centroids,labels,'NumNeighbors',1);

Dataclustering=model.predict(newdataset);

Rindex=randindex(Dataclustering,Reallabels)

%% analyzing the result (if the dataset is labeled)

for i=1: k
    
    numofsamples(i)=numel(find(Dataclustering==i));
    
end

numof_nonzero_clusters=numel(find(numofsamples>0))

for j=1:realclusternumber
    
    realnumofSam(j)=numel(find(Reallabels==j));
    
end

distance=zeros(k);

for i=1:k
    
    for j=1:k
        
        distance(i,j)=dist(centroids(i,:),centroids(j,:)');
        
    end
    
    
end

NMI=nmi(Reallabels,Dataclustering)
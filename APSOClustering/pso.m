%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Particle Swarm Optimization Alggorithm                         %
%                                                                %
% Programmed By: MatlabSite.com Programmers Team                 %
%                Copyright 2009                                  %
%                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;
close all

%% sorting the features in diversity

 numofclusters=15;


numofselectedfeatures=2;

nvar=numofclusters*numofselectedfeatures;
varsize=[1 nvar];

FuncIter=20;

% wholedataset=xlsread('poker');

% wholedataset=load('Mydataset9.mat');

dataset=xlsread('S1.xlsx');

% dataset=wholedataset(:,1:end-1);

% dataset=wholedataset.x;



Reallabels=xlsread('S1labels.xlsx');

% Realcenters=xlsread('Realcenters.xlsx');

% Label=L.label;

[~,numoffeatures]=size(dataset);
        for  i=1:numoffeatures


            data=dataset(:,i);

            div(i)=std(data);

        end


        div=div';
        
        [diversity featurenumber]=sort(div,'descend');
        
        [selectedfeatures newdataset]=RemovingFeatures(dataset,numofselectedfeatures);
        
        AVG=mean(newdataset);
%% num of population

% npop=round(sqrt(size(wholedataset,1)))-1;
npop=5;
        
 %%


w=1;
wdamp=0.99;

c1=2;
c2=2;
xmin=min(min(newdataset));
xmax=max(max(newdataset));
dx=xmax-xmin;

% xmin=min(newdataset);
% xmin1=xmin(1);
% xmin2=xmin(2);
% 
% xmax=max(newdataset);
% xmax1=xmax(1);
% xmax2=xmax(2);



vmax=0.1*dx;

maxit=50;

% d1=linspace(1.5,0,maxit);
% 
% C2=exp(2*d1);
% 
% % C1(round(0.9*maxit):end)=2;
% 
% d2=linspace(0,1.5,maxit);
% 
% C1=exp(2*d2);

% C2(round(0.9*maxit):end)=2;

empty_particle.position=[];
empty_particle.velocity=[];
empty_particle.cost=[];
empty_particle.pbest=[];
empty_particle.pbestcost=[];
empty_particle.centroids=[];
empty_particle.numofclusters=[];
% empty_particle.firstdim=[];
% empty_particle.seconddim=[];


particle=repmat(empty_particle,npop,1);

gbest=zeros(maxit,nvar);
gbestcost=zeros(maxit,1);
meanfits=zeros(maxit,1);

tic;

for it=1:maxit
    sumcost=0;
    
%       c1=C1(it);
%       c2=C2(it);
    if it==1
        gbestcost(1)=inf;
        for i=1:npop
            
            particle(i).velocity=zeros(1,nvar);
            
            particle(i).numofclusters=numofclusters;
            
            particle(i).position=rand*repmat(AVG,1,particle(i).numofclusters)+rand(1,particle(i).numofclusters*numofselectedfeatures)+randi([round(xmin) round(xmax)],1,nvar);
            
%             particle(i).position=zeros(1,nvar);

%             particle(i).firstdim = xmin1+(xmax1-xmin1)*rand(1,nvar/2);
            
%             particle(i).seconddim = xmin2+(xmax2-xmin2)*rand(1,nvar/2);
            
            particle(i).centroids=reshape(particle(i).position,numofselectedfeatures,[])';
            
%             particle(i).position(1:2:end)=particle(i).firstdim;
%             particle(i).position(2:2:end)=particle(i).seconddim;


%             particle(i).centroids = [particle(i).firstdim' particle(i).seconddim'];
                   
            particle(i).cost=fitness2(newdataset,particle(i).centroids,numofclusters);
            
%             if isnan(particle(i).cost)
%             
%                      [particle(i).position particle(i).centroids particle(i).cost]=revise(particle(i).cost,particle(i).position,nvar,numofselectedfeatures,numofclusters,newdataset);
%             
%             end
            
            particle(i).pbest=particle(i).position;
            particle(i).pbestcost=particle(i).cost;
            
            if particle(i).pbestcost<gbestcost(it)
                gbest(it,:)=particle(i).pbest;
                gbestcost(it)=particle(i).pbestcost;
            end
            sumcost=sumcost+particle(i).cost;
        end
        
    else
        gbest(it,:)=gbest(it-1,:);
        gbestcost(it)=gbestcost(it-1);
        for i=1:npop
            particle(i).velocity=w*particle(i).velocity...
                                +c1*rand*(particle(i).pbest-particle(i).position)...
                                +c2*rand*(gbest(it,:)-particle(i).position);

%             particle(i).velocity=w*particle(i).velocity...
%                                 +c1*(particle(i).pbest-particle(i).position)...
%                                 +c2*(gbest(it,:)-particle(i).position);
                            
            particle(i).velocity=min(max(particle(i).velocity,-vmax),vmax);
            
            particle(i).position=particle(i).position+particle(i).velocity;
            
            particle(i).position=min(max(particle(i).position,xmin),xmax);
            
%             particle(i).firstdim=min(max(particle(i).position(1:2:end),xmin1),xmax1);
            
%             particle(i).seconddim=min(max(particle(i).position(2:2:end),xmin2),xmax2);
            
%             particle(i).position(1:2:end)=particle(i).firstdim;
%             particle(i).position(2:2:end)=particle(i).seconddim;
            
            particle(i).centroids=reshape(particle(i).position,numofselectedfeatures,[])';
            
%             particle(i).centroids = [particle(i).firstdim' particle(i).seconddim'];
            

            
            particle(i).cost=fitness2(newdataset,particle(i).centroids,numofclusters);
            
%             if isnan(particle(i).cost)
%             
%                      [particle(i).position particle(i).centroids particle(i).cost]=revise(particle(i).cost,particle(i).position,nvar,numofselectedfeatures,numofclusters,newdataset);
%             
%             end
            
            if particle(i).cost<particle(i).pbestcost
                particle(i).pbest=particle(i).position;
                particle(i).pbestcost=particle(i).cost;

                if particle(i).pbestcost<gbestcost(it)
                    gbest(it,:)=particle(i).pbest;
                    gbestcost(it)=particle(i).pbestcost;
                end
            end
            sumcost=sumcost+particle(i).cost;
        end
        
    end
    
    meanfits(it)=sumcost/npop;
    
    disp(['Iteration ' num2str(it) ':   Best Cost = ' num2str(gbestcost(it)) '   average= ' num2str(meanfits(it))]);
    
    w=w*wdamp;
end

%% finding the best solution

globalbest.position=gbest(end,1:end);
globalbest.centroids=reshape(globalbest.position,numofselectedfeatures,[])';
globalbest.cost=gbestcost(end);

toc;

% ROC curve

plot(gbestcost,'b','Linewidth',3);
hold on
plot(meanfits,'r','Linewidth',3);

%% saving results

save('result.mat','globalbest');

%% plotting cluster centers

Centers=load('result.mat');

Centroids=Centers.globalbest.centroids;

% x1=dataset(1:100000,:);
% x2=dataset(100001:200000,:);
% x3=dataset(200001:300000,:);
% x4=dataset(300001:400000,:);
% x5=dataset(400001:500000,:);
% x6=dataset(500001:600000,:);
% x7=dataset(600001:700000,:);
% x8=dataset(700001:800000,:);
% x9=dataset(800001:900000,:);
% x10=dataset(900001:1000000,:);
% x11=dataset(10001:11000,:);
% x12=dataset(11001:12000,:);
% x13=dataset(12001:13000,:);
% x14=dataset(13001:14000,:);
% x15=dataset(14001:15000,:);


% figure;
% 
% plot(x1(:,1) ,x1(:,2), 'bo');
% hold on
% plot(x2(:,1) ,x2(:,2), 'sr');
% plot(x3(:,1) ,x3(:,2), 'kp');
% hold on
% plot(x4(:,1) ,x4(:,2), 'y+');
% hold on
% plot(x5(:,1) ,x5(:,2), 'dm');
% hold on
% plot(x6(:,1) ,x6(:,2), 'pc');
% hold on
% plot(x7(:,1) ,x7(:,2), 'x','Color',[128 0 0]/255);
% hold on
% plot(x8(:,1) ,x8(:,2), 'x','Color',[64 128 128]/255);
% hold on
% plot(x9(:,1) ,x9(:,2), 'x','Color',[255 128 0]/255);
% hold on
% plot(x10(:,1) ,x10(:,2), 'x','Color',[128 0 128]/255);
% hold on

figure

plot(dataset(:,1),dataset(:,2),'o');

hold on 

for j=1:numofclusters
    
    plot(Centroids(j,1),Centroids(j,2),'k*','markersize',10,'linewidth',1.5);
    hold on
    
end


% figure
% plot(C1,'linewidth',3);
% legend('C1');
% figure
% plot(C2,'linewidth',3);
% legend('C2');
% for j=1:numofclusters
%     
%     plot(Realcenters(j,1),Realcenters(j,2),'*','markersize',10,'linewidth',1.5);
%     hold on
%     
% end



% plot(Centroids(1,1),Centroids(1,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);
% hold on
% plot(Centroids(2,1),Centroids(2,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);
% hold on
% plot(Centroids(3,1),Centroids(3,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);
% hold on
% plot(Centroids(4,1),Centroids(4,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);
% hold on
% plot(Centroids(5,1),Centroids(5,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);
% hold on
% plot(Centroids(6,1),Centroids(6,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);
% hold on
% plot(Centroids(7,1),Centroids(7,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);
% hold on
% plot(Centroids(8,1),Centroids(8,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);
% hold on
% plot(Centroids(9,1),Centroids(9,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);
% hold on
% plot(Centroids(10,1),Centroids(10,2),'*','Color',[128 128 192]/255,'markersize',10,'linewidth',1.5);

%% calculating rand index

disp(' ');
disp('--------calculating the rand index-------');
disp(' ');

g=load('result.mat');
     
centroids=g.globalbest.centroids;
  
labels=[1:size(centroids,1)]';
  
% Dataclustering=knnclassify(newdataset,centroids,labels);

model=fitcknn(centroids,labels,'NumNeighbors',1);

Dataclustering=model.predict(newdataset);
  
Rindex=randindex(Dataclustering,Reallabels)











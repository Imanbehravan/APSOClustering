function [ G ] = PSOfunc( numofselectedfeatures,numofclusters,newdataset,maxit,npop,population,numofrepres,xmin,xmax )



nvar=numofclusters*numofselectedfeatures;
%varsize=[1 nvar];

        
% AVG=mean(newdataset);


        
 %%


w=1;
wdamp=0.99;

% c1=2;
% c2=3;


c1=2;
c2=2;


%xmin=min(min(newdataset));
%xmax=max(max(newdataset));
dx=xmax-xmin;

% xmin=min(newdataset);
% xmin1=xmin(1);
% xmin2=xmin(2);
% 
% xmax=max(newdataset);
% xmax1=xmax(1);
% xmax2=xmax(2);



vmax=0.1*dx;



empty_particle.position=[];
empty_particle.velocity=[];
empty_particle.cost=[];
empty_particle.pbest=[];
empty_particle.pbestcost=[];
empty_particle.centroids=[];
empty_particle.numofclusters=[];
empty_particle.inspectedK=[];
% empty_particle.firstdim=[];
% empty_particle.seconddim=[];


particle=repmat(empty_particle,npop,1);

gbest=zeros(maxit,nvar);
gbestcost=zeros(maxit,1);
gbestinspectedK=zeros(maxit,1);
meanfits=zeros(maxit,1);

for it=1:maxit
    sumcost=0;
    if it==1
        gbestcost(1)=inf;
        for i=1:npop
            
            particle(i).velocity=zeros(1,nvar);
            
            particle(i).numofclusters=numofclusters;
            
%             particle(i).position=rand*repmat(AVG,1,particle(i).numofclusters)+rand(1,particle(i).numofclusters*numofselectedfeatures)+randi([round(xmin) round(xmax)],1,nvar);
            
            particle(i).position=population(i,:);
            
%             particle(i).position=zeros(1,nvar);

%             particle(i).firstdim = xmin1+(xmax1-xmin1)*rand(1,nvar/2);
            
%             particle(i).seconddim = xmin2+(xmax2-xmin2)*rand(1,nvar/2);
            
            particle(i).centroids=reshape(particle(i).position,numofselectedfeatures,[])';
            
%             particle(i).position(1:2:end)=particle(i).firstdim;
%             particle(i).position(2:2:end)=particle(i).seconddim;


%             particle(i).centroids = [particle(i).firstdim' particle(i).seconddim'];
                   
            C=fitness6(numofselectedfeatures,numofclusters,newdataset,particle(i).centroids);
            
            particle(i).cost=1/(C.CriterionValues);
            
            particle(i).inspectedK=C.InspectedK;
            
            
            
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
                gbestinspectedK(it)=particle(i).inspectedK;
            end
            sumcost=sumcost+particle(i).cost;
        end
        
    else
        gbest(it,:)=gbest(it-1,:);
        gbestcost(it)=gbestcost(it-1);
        gbestinspectedK(it)=gbestinspectedK(it-1);
        for i=1:npop
            particle(i).velocity=w*particle(i).velocity...
                                +c1*rand*(particle(i).pbest-particle(i).position)...
                                +c2*rand*(gbest(it,:)-particle(i).position);

%             particle(i).velocity=w*particle(i).velocity...
%                                 +c1*(particle(i).pbest-particle(i).position)...
%                                 +c2*(gbest(it,:)-particle(i).position);
                            
            particle(i).velocity=min(max(particle(i).velocity,-vmax),vmax);
            
            particle(i).position=particle(i).position+particle(i).velocity;
            
%             particle(i).position=min(max(particle(i).position,xmin),xmax);
            
%             particle(i).firstdim=min(max(particle(i).position(1:2:end),xmin1),xmax1);
            
%             particle(i).seconddim=min(max(particle(i).position(2:2:end),xmin2),xmax2);
            
%             particle(i).position(1:2:end)=particle(i).firstdim;
%             particle(i).position(2:2:end)=particle(i).seconddim;
            
            particle(i).centroids=reshape(particle(i).position,numofselectedfeatures,[])';
            
            particle(i).centroids=AssingningNearestSamp(particle(i).centroids,newdataset,numofrepres,numofclusters);
            
            particle(i).position=reshape([particle(i).centroids]',1,numofclusters*numofselectedfeatures);
            
%             particle(i).centroids = [particle(i).firstdim' particle(i).seconddim'];
            

            
            C=fitness6(numofselectedfeatures,numofclusters,newdataset,particle(i).centroids);
            
            particle(i).cost=1/(C.CriterionValues);
            
            particle(i).inspectedK=C.InspectedK;
            
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
                    gbestinspectedK(it)=particle(i).inspectedK;
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
globalbest.inspectedK=gbestinspectedK(end);

% ROC curve

% plot(gbestcost,'b','Linewidth',3);
% hold on
% plot(meanfits,'r','Linewidth',3);

% %% saving results
% 
% save('result.mat','globalbest');
% 
% %% plotting cluster centers
% 
% Centers=load('result.mat');
% 
% Centroids=Centers.globalbest.centroids;



G=globalbest;










end


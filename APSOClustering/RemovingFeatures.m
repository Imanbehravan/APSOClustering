function [ selectedfeatures newdataset ] = RemovingFeatures( dataset,numofselectedfeatures )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


        [~,numoffeatures]=size(dataset);
        for  i=1:numoffeatures


            data=dataset(:,i);

            div(i)=std(data);

        end


        div=div';
        
        [diversity featurenumber]=sort(div,'descend');
        
        selectedfeatures=sort(featurenumber(1:numofselectedfeatures));
        
        for i=1:numel(selectedfeatures)
    
            data2(:,i)=dataset(:,selectedfeatures(i));

        end
        
        newdataset=data2;


end


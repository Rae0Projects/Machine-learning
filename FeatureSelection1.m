% Loading and reading the excel heart dataset file
Heart_Disease = readtable('Heartdataset.csv');
% Summarising the data and cleaning it
summary(Heart_Disease)
Heart_DiseaseClean = rmmissing(Heart_Disease)
size(Heart_Disease)
size(Heart_DiseaseClean)
summary(Heart_DiseaseClean)

% Transforming the table into an array
X = table2array(Heart_Disease(:,1:13));
Y = table2array(Heart_Disease(:,14));
 
subplot(4,4,1)
scatter(X(:,1),Y)
subplot(4,4,2)
scatter(X(:,2),Y)
subplot(4,4,3)
scatter(X(:,3),Y)
subplot(4,4,4)
scatter(X(:,4),Y)
subplot(4,4,5)
scatter(X(:,5),Y)
subplot(4,4,6)
scatter(X(:,6),Y)
subplot(4,4,7)
scatter(X(:,7),Y)
subplot(4,4,8)
scatter(X(:,8),Y)
subplot(4,4,9)
scatter(X(:,9),Y)
subplot(4,4,10)
scatter(X(:,10),Y)
subplot(4,4,11)
scatter(X(:,11),Y)
subplot(4,4,12)
scatter(X(:,12),Y)
subplot(4,4,13)
scatter(X(:,13),Y)

Model_1 = stepwiselm(X,Y,'constant','ResponseVar','ResResistance')
Model_2 = stepwiselm(X,Y,'linear','ResponseVar','ResResistance')
Model_3 = stepwiselm(X,Y,'interactions','ResponseVar','ResResistance')
Model_4 = stepwiselm(X,Y,'quadratic','ResponseVar','ResResistance','Upper','quadratic')

Rsquared = [Model_1.Rsquared.Adjusted,Model_2.Rsquared.Adjusted,Model_3.Rsquared.Adjusted,Model_4.Rsquared.Adjusted]

subplot(2,2,1)
plotResiduals(Model_1,'fitted')
subplot(2,2,2)
plotResiduals(Model_2,'fitted')
subplot(2,2,3)
plotResiduals(Model_3,'fitted')
subplot(2,2,4)
plotResiduals(Model_4,'fitted')

% Randomly splitting the dataset into training and test data
% (80%) = train and (20%) = test
rng(100)
Data_split = cvpartition(size(Heart_Disease,1),'Holdout',0.2);
test_data = Data_split.test;

% Split dataset into training and test data
Heart_Disease_train_split = Heart_Disease(~test_data,:);
Heart_Disease_test_split = Heart_Disease(test_data,:);

% Dividing the dataset into features and labels
Heart_Disease_train_split_labels = Heart_Disease_train_split(:,14);
Heart_Disease_train_split_features = Heart_Disease_train_split(:,1:13);
Heart_Disease_test_split_labels = Heart_Disease_test_split(:,14);
Heart_Disease_test_split_features = Heart_Disease_test_split(:,1:13);

figure()
disp(Heart_Disease_train_split_labels)
disp(Heart_Disease_train_split_features)

[test_data,tally] = fscmrmr(Heart_Disease_train_split_features,Heart_Disease_train_split_labels);
h = bar(tally(test_data))

xlabel('Rank of the predictor')
ylabel('The predictor importance score')
XtickLabels = Heart_Disease_train_split_features.Properties.VariableNames(test_data);
xtickangle(45)
title('Evaluating the importance of each feature ')




		

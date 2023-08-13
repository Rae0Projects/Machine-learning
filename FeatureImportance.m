% Loading and reading the excel heart dataset file
Heart_Disease = readtable('Heartdataset.csv');

% Converting logical variables into categorical arrays.
cat_columns = {'age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'}
Cat_columns_conversion = ismember(Heart_Disease.Properties.VariableNames, cat_columns);
for i = 1:length(cat_columns)
    Heartcolumns = cat_columns{i};
    Heart_Disease.(Heartcolumns) = categorical(Heart_Disease{:,Heartcolumns});
end
disp(Heart_Disease.(Heartcolumns))

% Assigning scores to input features to a predictive model that rates its
% importance.
% Changing the range and the shape of the distribution of the data using
% scaling and normalisation

% Converting logical variables into categorical arrays.
cat_columns = {'age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'}
Cat_columns_conversion = ismember(Heart_Disease.Properties.VariableNames, cat_columns);
for i = 1:length(cat_columns)
    Heartcolumns = cat_columns{i};
    Heart_Disease.(Heartcolumns) = categorical(Heart_Disease{:,Heartcolumns});
end
disp(Heart_Disease.(Heartcolumns))

% Displaying normalization
disp(normalizedData_Age);
disp(normalizedData_Sex);
disp(normalizedData_Cp);
disp(normalizedData_Trestbps);
disp(normalizedData_Chol);
disp(normalizedData_Fbs);
disp(normalizedData_Restecg);
disp(normalizedData_Thalach);
disp(normalizedData_Exang);
disp(normalizedData_OldPeak);
disp(normalizedData_Ca);
disp(normalizedData_Thal);

% Randomly splitting the dataset into training and test data
% (80%) = train and (20%) = test
rng(100)
Data_split = cvpartition(size(Heart_Disease,1),'Holdout',0.2);
test_data = Data_split.test;

% Split dataset into training and test data
Heart_Disease_train_split = Heart_Disease(~test_data,:);
Heart_Disease_test_split = Heart_Disease(test_data,:);

% Reducing the training dataset to a more managable size using
% downsampling.
hist_bin_counts = histcounts(Heart_Disease_train_split.target);
fprintf('There are %d 0s and %d 1s in the training dataset for target,\n',hist_bin_counts(1),hist_bin_counts(2))
Heart_1_rows = find(Heart_Disease_train_split.target=='1');
Heart_0_rows = find(Heart_Disease_train_split.target=='0');
downsampling = Heart_0_rows(1:length(Heart_1_rows));
Extracting_rows = sort([downsampling;Heart_1_rows]);
Heart_Disease_train_split = Heart_Disease_train_split(Extracting_rows,:);
Count_Downsampling = histcounts(Heart_Disease_train_split.target);

% Dividing the dataset into features and labels
Heart_Disease_train_split_labels = Heart_Disease_train_split(:,14);
Heart_Disease_train_split_features = Heart_Disease_train_split(:,1:13);
Heart_Disease_test_split_labels = Heart_Disease_test_split(:,14);
Heart_Disease_test_split_features = Heart_Disease_test_split(:,1:13);

figure()
disp(Heart_Disease_train_split_labels)
disp(Heart_Disease_train_split_features)

% MRMR (minimum redundancy maximum relevance) to rank features and score
% index, visually displayed using a bar chart
[test_data,tally] = fscmrmr(Heart_Disease_train_split_features,Heart_Disease_train_split_labels);

bar(tally(test_data))
xlabel('Rank of the predictor')
ylabel('The predictor importance score')
XtickLabels = Heart_Disease_train_split_features.Properties.VariableNames(test_data);
xtickangle(45)

% Random forest hyperparameter tuning. Please wait, this may take a couple
% of minutes to run.
























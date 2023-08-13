% Random forest hyperparameter tuning. Please wait, this may take a couple
% of minutes to run.

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

% Creating variables
Heart_Age = Heart_Disease.age;
disp(Heart_Age)
Heart_Sex = Heart_Disease.sex;
Heart_Cp = Heart_Disease.cp;
Heart_Trestbps = Heart_Disease.trestbps;
Heart_Chol = Heart_Disease.chol;
Heart_Fbs = Heart_Disease.fbs;
Heart_Restecg = Heart_Disease.restecg;
Heart_Thalach = Heart_Disease.thalach;
Heart_Exang = Heart_Disease.exang;
Heart_Oldpeak = Heart_Disease.oldpeak;
Heart_Slope = Heart_Disease.slope;
Heart_Ca = Heart_Disease.ca;
Heart_Thal = Heart_Disease.thal;

% Normalize data task using live script
% Normalize Data
normalizedData_Age = normalize(Heart_Age,"center","scale");
normalizedData_Sex = normalize(Heart_Sex,"center","scale");
normalizedData_Cp = normalize(Heart_Cp,"center","scale");
normalizedData_Trestbps = normalize(Heart_Trestbps,"center","scale");
normalizedData_Chol = normalize(Heart_Chol,"center","scale");
normalizedData_Fbs = normalize(Heart_Fbs,"center","scale");
normalizedData_Restecg = normalize(Heart_Restecg,"center","scale");
normalizedData_Thalach = normalize(Heart_Thalach,"center","scale");
normalizedData_Exang = normalize(Heart_Exang,"center","scale");
normalizedData_OldPeak = normalize(Heart_Oldpeak,"center","scale");
normalizedData_Slope = normalize(Heart_Slope,"center","scale");
normalizedData_Ca = normalize(Heart_Ca,"center","scale");
normalizedData_Thal = normalize(Heart_Thal,"center","scale");

% Randomly splitting the dataset into training and test data
% (80%) = train and (20%) = test
rng(100)
Data_split = cvpartition(size(Heart_Disease,1),'Holdout',0.2);
test_data = Data_split.test;

% Split dataset into training and test data
Heart_Disease_train_split = Heart_Disease(~test_data,:);
Heart_Disease_test_split = Heart_Disease(test_data,:);

% Downsample the training data
training_histcount = histcounts(Heart_Disease_train_split.target);
fprint(' There are %d 0s and %d 1s for target in the training dataset.\n',training_histcount(1),training_histcount(2))
HeartDisease_1sRows = find(Heart_Disease_train_split.target=='1');
HeartDisease_0sRows = find(Heart_Disease_train_split.target=='0');
downsampled = HeartDisease_0sRows(1:length(HeartDisease_1sRows));
Extracted_rows = sort([downsampled;HeartDisease_1sRows]);
Heart_Disease_train_split = Heart_Disease_train_split(Extracted_rows,:);
downsample_histcount = histcount(Heart_Disease_train_split.target);
fprintf('There are %d 0s and %d 1s in the downsampled training dataset for target.\n',downsample_histcount(1),downsample_histcount(2))

% Dividing the data into lables and features split
% Splitting the data into features and labels
Heart_disease_train_labels = Heart_disease_train(:,14);
Heart_disease_train_features = Heart_disease_train(:,1:13);

Heart_disease_test_labels = Heart_disease_test(:,14);
Heart_disease_test_features = Heart_disease_test(:,1:13);















% Confusion matrix for Naive Bayes
figure()
NaiveBayes_Confusion_Matrix = confusionmat(Heart_disease_test_labels,Group);
result = confusionchart(Heart_disease_test_labels,Group)
NBcm = NaiveBayes_Confusion_Matrix'

diagonalNB = diag(NBcm)
sum_of_rows_NB = sum(NBcm, 2)

precisionNB = diagonalNB ./ sum_of_rows_NB
overall_precisionNB = mean(precisionNB)

sum_of_columnsNB = sum(NBcm, 1)

recallNB = diagonalNB ./ sum_of_columnsNB'
overall_recallNB = mean(recallNB)

f1_scoreNB = 2*((overall_precisionNB*overall_recallNB)/(overall_precisionNB+overall_recallNB))

Accuracy_NB = 100*sum(diag(NBcm))./sum(NBcm(:));  
disp(Accuracy_NB)

TP_NB = 15;
FN_NB = 6;
FP_NB = 11;
TN_NB = 28;

specificityNB = (TN_NB./(TN_NB+FP_NB))

% Reading from confusion matrix chart:


fprintf("Naive Bayes performance metrics\n")
fprintf("Naive Bayes accuracy: %f\n",Accuracy_NB )
fprintf("Naive Bayes specificity: %f\n",specificityNB)
fprintf("Naive Bayes precision: %f\n", overall_precisionNB)
fprintf("Naive Bayes recall: %f\n",overall_recallNB)
fprintf("Naive Bayes F1 score: %f\n",f1_scoreNB)

title("Naive Bayes classifier confusion matrix")










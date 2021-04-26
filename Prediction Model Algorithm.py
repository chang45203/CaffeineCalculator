# Importing Required Packages
import os
import pandas as pd
import openpyxl
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

os.listdir() # gives a list containing the names of all the entries in the path
pd.options.display.max_rows = 30  # allows Pycharm to see the whole dataset
pd.options.display.max_columns = 30
pd.options.display.width=None

# Load the Data
original_df = pd.read_excel("/Users/seanchang/Desktop/Jupyter Notebook Stuff/Master Sheet Caffeine Metabolism "
                            "Research .xlsx", sheet_name = 'All Data')
original_df.tail(3)  # use to check whether you can see full DF

# Clean the Data
exp_group_df = original_df.loc[original_df['Est. HR Recovery']!=0]  # Remove the Control Subjects from the DF

    # Replacing all M/F as Integers --> (M = 0, F = 1)
# exp_group_df['Gender'] = [0 if x== 'M' else 1 for x in exp_group_df['Gender']]  # Albert's Code
exp_group_df.loc[exp_group_df['Gender'] == 'M', 'Gender'] = 0
exp_group_df.loc[exp_group_df['Gender'] == 'F', 'Gender'] = 1

exp_group_df.shape

# Exploratory Data Analysis (EDA)
    # describe.() method shows general stats of each variable
exp_group_df.describe()

# Plotting the EDA Graph
# sns.pairplot(exp_group_df[['Est. HR Recovery','Bodyweight (kg)','Basal Metabolic Rate','Height (cm)','Gender']])
# plt.show()
        # It seems the HR recovery time and the independent variables we suspect have strong linear relationship.
        # This implies classic linear regression, or OLS (ordinary least square) can perform well for our analysis.
        # We see there's one outlier that has exceptionally fast HR recovery time (<100) with low height and weight.
        # Removing that one outlier can significantly improve the model fit to the data.

# Modeling
    # EDA suggests that heart rate recovery time is linearly correlated with other variables, such as Bodyweight, Basal Metablic Rate, and Height.
    # Thus linear regression can be a good model candidate.

# erase the outlier plot we discovered last time
sample_clean_df = exp_group_df.loc[exp_group_df['Phone Number']!=81293460]

# Plot the data distribution of the remaining subjects
# plt.scatter(sample_clean_df['Bodyweight (kg)'],sample_clean_df['Est. HR Recovery'])
# plt.xlabel('Bodyweight (kg)'); plt.ylabel('Est. HR Recovery'); plt.title('Data Distribution btw HR recovery time '
#                                                                          'vs. Body weight')
# plt.show()

# Convex Optimization
# X2 = sm.add_constant(sample_clean_df['Bodyweight (kg)'])
# est = sm.OLS(sample_clean_df['Est. HR Recovery'],X2)
# est = est.fit()
# ypred = est.predict(X2)

# Plotting the Convex Optimization
# plt.scatter(sample_clean_df['Bodyweight (kg)'],sample_clean_df['Est. HR Recovery'])
# plt.plot(sample_clean_df['Bodyweight (kg)'], ypred, c = 'red')
# plt.xlabel('Bodyweight (kg)');plt.ylabel('Est. HR Recovery');plt.title('Data Distribution '
#                                                                        'btw HR recovery time vs. Body weight')
# plt.show()
    # While we showed a very clean, linear correlation in the example above, we actually removed an outlier before doing it.
    # If you plot the full experiment group's data distribution, there's an outlier at the bottom left

# Modeling Before Excluding the Outlier
# plt.scatter(exp_group_df['Bodyweight (kg)'],exp_group_df['Est. HR Recovery'])
# plt.xlabel('Bodyweight (kg)'); plt.ylabel('Est HR Recovery'); plt.title('Data Distribution btw '
#                                                                         'HR recovery time vs. Body weight')
# plt.show()

# Model Fit Before Excluding The Outlier
# X0 = exp_group_df[['Height (cm)', 'Bodyweight (kg)','Basal Metabolic Rate', 'Gender']]
# y0 = exp_group_df['Est. HR Recovery']
#
# X0 = sm.add_constant(X0)
# est = sm.OLS(y0.astype(float), X0.astype(float))
# est = est.fit()
# print(est.summary())

# Model Fit After Excluding The Outlier
    # Although we have limited data set (11 obs), removing the outlier (Phone #: 81293460) makes the coefficient values
    # and model fit reasonable

X = sample_clean_df[['Height (cm)','Bodyweight (kg)','Basal Metabolic Rate','Gender']]
y = sample_clean_df['Est. HR Recovery']

# X2 = sm.add_constant(X)
# est = sm.OLS(y.astype(float),X2.astype(float))
# est = est.fit()
# print(est.summary())

    # Findings:
        # We see that the model fit, measured by adjusted R-squred, has greatly improved from ~0 to 0.86 after removing the outlier. Thus the outlier removal can be justified based on this.
        # It shows somewhat counter-intuitive result - now, Height and basic metabolic rate are not significant
            # - even though our pairwise plot shows otherwise.
            # A part of the reasons is the different scales within our data.
            # Since the scale of our regressors (height, weight, metabolism rate, gender) are different, independent variables with highest magnitude/range would show the highest influence in our prediction.
        # Alternatively you can do normalization instead.
            # Since we expect Gaussian distribution is common, such as in weight and height, we can choose standardization here.

# Standardization
    # Standardization is needed to make sure their correlation values are properly done. Currently, height (cm) and bodyweight (kg) units confuse the data.

feature_mean = X.mean()
# print(feature_mean)
feature_std = X.std()
# print(feature_std)

X_standardized = pd.DataFrame()
for idx, col in enumerate(X.columns.tolist()):
    X_standardized[col] = (X[col]-feature_mean[idx])/feature_std[idx]


y_mean = y.mean()
y_std = y.std()
y_standardized = (y-y_mean)/y_std

X2 = sm.add_constant(X_standardized)

est = sm.OLS(y_standardized.astype(float), X2.astype(float))
est = est.fit()
# print(est.summary())
    # This est is HAVING PROBLEMS. DOESN'T RUN FULLY.
    # Standard Errors assume that the covariance matrix of the errors is correctly specified.
    # It seems better now; the effect of Gender and constant have materially decreased after standardization.
    # However, the effect of body weight and height is somewhat suspicous, given that there is a high correlation between those and metabolic rate.

# Collinearity Check
# sns.pairplot(pd.concat([y_standardized, X_standardized], axis = 1))
# plt.show()

# Correlation Heatmap
# sns.heatmap(sample_clean_df[['Est. HR Recovery','Height (cm)','Bodyweight (kg)',
#                              'Basal Metabolic Rate','Gender']].corr(),cmap='bwr',annot = True)
# plt.show()
    # Not surprisingly, our regressors are strongly correlated with each other. we can solve it in two ways:
            # either use interaction term of two (or more) or
            # drop regressor(s) we believe redundant
    # Here we will take the second approach, since regressors show strong linear relationships (e.g. BMR vs. bodyweight), thus some of them are replaceable.
    # Due to small sample size, stepwise regression seems to be the best option here (lasso requiring larger dataset for cross validation)

# Forward Stepwise Regression
for i in X_standardized.columns.tolist():
    X2 = X_standardized[i]
    est = sm.OLS(y_standardized.astype(float), X2.astype(float))
    est = est.fit()
    # print("%s:  %s" % (i, round(est.rsquared_adj, 3)))

# Bodyweight has the highest adjusted R squared value, so select the variable first.
for i in X_standardized.drop(columns = ['Bodyweight (kg)']).columns.tolist():
    X2 = X_standardized[['Bodyweight (kg)',i]]
    est = sm.OLS(y_standardized.astype(float), X2.astype(float))
    est = est.fit()
    # print("%s:  adj Rsq: %s" % (i, round(est.rsquared_adj, 3)))

# Metabolic Rate
for i in X_standardized.drop(columns = ['Bodyweight (kg)','Basal Metabolic Rate']).columns.tolist():
    X2 = X_standardized[['Bodyweight (kg)','Basal Metabolic Rate',i]]
    est = sm.OLS(y_standardized.astype(float),X2.astype(float))
    est = est.fit()
    # print( "%s:  %s" %  (i, round(est.rsquared_adj, 3)))

# Gender
for i in X_standardized.drop(columns = ['Height (cm)']).columns.tolist():
    est = sm.OLS(y_standardized.astype(float), X2.astype(float))
    est = est.fit()
    # print(est.summary())
        # We see height actually decrease adjusted R-squred (to 0.86).
        # We've achieved the highest adjusted R squared. so let's settle here.
        # Note that bodyweight indeed is no longer statistically significant.
            # This is possible due to collinearity we could not remove due to the nature of the data.
        # A solution to this is using backward stepwise regression that removes all regressors if p value greater than 0.05
        # Turns out, removing bodyweight factor makes the model as concise and reasonably fitted to the data

# Seeing influence of BMR and Gender
X2 =X_standardized.drop(columns = ['Height (cm)', 'Bodyweight (kg)'])
est = sm.OLS(y_standardized.astype(float), X2.astype(float))
est = est.fit()
# print(est.summary())
    # We see that BMR has a higher effect than Gender

# Prediction Model
y_pred = est.predict(X2)
y_pred_rescaled = y_pred * y_std * y_mean
plt.scatter(sample_clean_df['Est. HR Recovery'], y_pred_rescaled)
plt.plot([0,500],[0,500], c = 'red')
plt.ylim([0,400])
plt.xlim([0,400])
plt.xlabel('Actual HR Recovery Time (min)')
plt.ylabel('Predicted HR Recovery Time (min)')

plt.show()

# Error Plot
plt.scatter(range(y_pred_rescaled.shape[0]),sample_clean_df['Est. HR Recovery'] - y_pred_rescaled)
plt.plot([-100, 100], [0, 0], c = 'red')
plt.xlim([-1,10])
plt.xlabel('Index')
plt.ylabel('Difference in Actual vs. Predicted')

# plt.show()

# Conclusion
# Our analysis suggests that metabolic rate and gender are the two indicators that can explain the Heart Rate recovery time within the normalized dataset. i.e.
# 1 standard deviation increase in BMR leads to 1.6 standard deviation decrease in heart rate recovery.
# 1 standard deviation increase in Gender (roughly speaking, if a subject is a woman) leads to 0.8 standard deviation decrease in heart rate recovery time
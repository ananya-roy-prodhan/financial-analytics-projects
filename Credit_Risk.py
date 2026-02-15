import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from websockets.legacy.framing import encode_data
import statsmodels.api as sm

# inputs
credit_data = pd.read_csv(r"C:\Users\Annie\Documents\Study Materials\HKU Finance and Data Analysis\Financial Data Analytics with Python\Project\credit_risk_dataset.csv")
show_plots = True
save_plots = False
streamlit_dump = False

# Investigating Data quality
credit_data.info()
print(credit_data.isnull().values.any())
has_missing = credit_data.isnull().any()
col_with_missing_list = credit_data.columns[has_missing].tolist()
print(col_with_missing_list)
#blank_rows = credit_data[credit_data[['person_emp_length']].isnull().any(axis=1)] # axis =1 represents rows
#len(blank_rows)
#rows_with_no_blank = credit_data.dropna().shape[0]
#data_col = credit_data.columns


#Correlation Analysis
# Checked the correlation of columns with missing values with rest of the columns
# Separated the numeric and categorical columns
credit_data_numeric = credit_data.select_dtypes(include=['int64', 'float64'])
credit_data_cat = credit_data.select_dtypes(include=['object', 'category'])
# numeric correlation matrix
credit_data_corr_matrix = credit_data_numeric.corr()

# Impute missing value of loan_int_rate, as it can be highly correlated with loan grade
credit_data_new = credit_data.copy()
median_rate_by_grade = credit_data_new.groupby('loan_grade')['loan_int_rate'].median()
# print and see if the rates are indeed correlated with grade
print(median_rate_by_grade)
# since we can see string correlation, we should replace the missing rates by respective grade median
credit_data_new['loan_int_rate'] = credit_data_new.apply(
    lambda row: median_rate_by_grade[row['loan_grade']]
                if pd.isna(row['loan_int_rate'])
                else row['loan_int_rate'],
    axis=1)


# Removing blank data in person_emp_length
df = credit_data_new.dropna()



# Data Preparation: Outlier Detection

# Box plot
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
for var in numeric_cols.columns:
    plt.figure(figsize=(12, 6))
    flierprops = dict(marker='o', markersize=12, markerfacecolor='red', linestyle='none')
    sns.boxplot(df[var], flierprops=flierprops)
    plt.title(f"Box Plot of {var}", fontsize=22, fontweight='bold')
    if save_plots:
        plt.savefig(f"Box Plot of {var}.png")

# def remove_outlier_IQR(df_input):
#     numeric_cols = df_input.select_dtypes(include=['int64', 'float64'])
#     numeric_colnames = list(numeric_cols.columns)
#     numeric_colnames.remove("loan_status")
#     df_clean = df_input.copy()
#     for col in numeric_colnames:
#         Q1 = df_input[col].quantile(0.25)
#         Q3 = df_input[col].quantile(0.75)
#         IQR = Q3 - Q1
#         Lower = Q1 - 1.5 * IQR
#         Upper = Q3 + 1.5 * IQR
#         print(f"Variable {col} has Lower outlier limit of {Lower} and Upper outlier limit of {Upper}")
#         n_row_og = df_clean.shape[0]
#         df_clean = df_clean[(df_clean[col] >= Lower) & (df_clean[col] <= Upper)]
#         n_row_after = df_clean.shape[0]
#         n_row_removed = n_row_og - n_row_after
#         print(f"Number of rows removed {n_row_removed} after removing outlier from {col}")
#     return df_clean

# Outlier Removal
def remove_outlier_zscore(df_input, z_score=3):
    numeric_cols = df_input.select_dtypes(include=['int64', 'float64'])
    numeric_colnames = list(numeric_cols.columns)
    numeric_colnames.remove("loan_status")
    df_clean = df_input.copy()
    for col in numeric_colnames:
        std_dev = df_input[col].std()
        mean = df_input[col].mean()
        Lower = mean - z_score * std_dev
        Upper = mean + z_score * std_dev
        print(f"Variable {col} has Lower outlier limit of {Lower} and Upper outlier limit of {Upper}")
        n_row_og = df_clean.shape[0]
        df_clean = df_clean[(df_clean[col] >= Lower) & (df_clean[col] <= Upper)]
        n_row_after = df_clean.shape[0]
        n_row_removed = n_row_og - n_row_after
        print(f"Number of rows removed {n_row_removed} after removing outlier from {col}")
    return df_clean

df_no_outlier = remove_outlier_zscore(df,10)

# df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
# Pair plot
# numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
# df_numeric = df[numeric_cols].copy()
# df_numeric['loan_status'] = df['loan_status']
# pair = sns.pairplot(df_numeric, hue="loan_status", diag_kind="hist", palette="tab10", plot_kws={"s": 20})
# for ax in pair.axes.flatten():
#     if ax is not None:
#         ax.set_xlabel(ax.get_xlabel(), fontsize=12)
#         ax.set_ylabel(ax.get_ylabel(), fontsize=12)
# plt.suptitle("Pair Plot of Numeric Loan Variables", y=1.02)
# plt.savefig("Pair Plot.png")
# if show_plots:
#   plt.show()


# Exploratory Data Analysis
# 1. Dist of borrower age
plt.figure(figsize=(10,6))
sns.histplot(df_no_outlier['person_age'], bins=20, kde=True, color='slateblue')
plt.title("Distribution of Borrower Ages", fontsize=20, fontweight='bold')
plt.xlabel("Age", fontsize=14)
plt.ylabel("Number of Borrowers", fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
plt.axvline(df_no_outlier['person_age'].median(), color='red', linestyle='--', label=f"Median Age: {df['person_age'].median():.1f}")
plt.legend(fontsize=14, title_fontsize=14)
if save_plots:
    plt.savefig("Histogram of Ages.png", dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()

# 2.Comparison of income between defaulters vs. non-defaulters
plt.figure(figsize=(10,6))
sns.boxplot(x ='loan_status', y = 'person_income', hue='loan_status', data =df_no_outlier,palette={0: "slateblue", 1: "red"}, legend=False)
plt.title("Income Comparison: Defaulters vs Non-Defaulters", fontsize=20, fontweight='bold')
plt.xlabel("Delinquency", fontsize=14)
plt.ylabel("Income", fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
if save_plots:
    plt.savefig("Income Comparison.png", dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()

# 3.Comparison of loan to income ratio between defaulters vs. non-defaulters
plt.figure(figsize=(10,6))
sns.boxplot(x ='loan_status', y = 'loan_percent_income', hue='loan_status', data =df_no_outlier,palette='tab10', legend=False)
plt.title("Loan as % of Income Comparison: Defaulters vs Non-Defaulters", fontsize=18, fontweight='bold')
plt.xlabel("Delinquency", fontsize=14)
plt.ylabel("Loan/Income", fontsize=14)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
if save_plots:
    plt.savefig("Loan as % of Income Comparison.png", dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()

# # 3. Hist of income levels
# plt.figure(figsize=(10,6))
# sns.histplot(encoded_df['person_income'], bins=30, kde=True, color='blue')
# plt.title("Histogram of Borrower Annual Income", fontsize=20, fontweight='bold')
# plt.xlabel("Annual Income")
# plt.ylabel("Number of Borrowers")
# if show_plots:
#     plt.show()

# 4.Proportion of borrowers by home ownership
plt.figure(figsize=(10,6))
sns.countplot(x='person_home_ownership', hue='loan_status', data=df_no_outlier, palette={0: "indigo", 1: "red"})
plt.title("Home Ownership vs Default Risk", fontsize=18, fontweight='bold')
plt.xlabel("Home Ownership", fontsize=14)
plt.ylabel("Number of Borrowers", fontsize=14)
plt.legend(title="Delinquency", title_fontsize=14)
if save_plots:
    plt.savefig("Home Ownership vs Default Risk indigo.png", dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()

# 5. Interest rates and default probability
df_rate_bins = df_no_outlier.copy()
df_rate_bins['rate_bin'] = pd.cut(df_no_outlier['loan_int_rate'], bins =10)
default_prob = (df_rate_bins.groupby('rate_bin', observed =True)['loan_status'].mean().reset_index())
plt.figure(figsize=(10,8))
sns.set_style("white")
ax = plt.gca()
ax.set_facecolor("#F1EFFE")
sns.lineplot(x=default_prob['rate_bin'].astype(str), y=default_prob['loan_status'], color = "indigo", marker='o', markersize=10)
plt.title("Default Probability Across Interest Rate Ranges", fontsize=18, fontweight='bold')
plt.xlabel("Interest Rate Range", fontsize=14)
plt.ylabel("Probability of Default", fontsize=14)
plt.xticks(rotation=45)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
if save_plots:
    plt.savefig("Interest Rate vs Default Risk.png", dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()


# 6. Loan intent dist
loan_counts = df_no_outlier['loan_intent'].value_counts()
colors = sns.color_palette("mako", len(loan_counts))
plt.figure(figsize=(9,9))
wedges, texts, autotexts = plt.pie(loan_counts, labels= None, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(width=0.4))
# To shift the %ages in the middle of each wedge
inner_radius = 1 - 0.4   # = 0.6
outer_radius = 1.0
mid_radius = (inner_radius + outer_radius) / 2
for i, autotext in enumerate(autotexts):
    angle = (wedges[i].theta2 + wedges[i].theta1) / 2  # Convert to radians
    x = mid_radius * np.cos(np.deg2rad(angle))
    y = mid_radius * np.sin(np.deg2rad(angle))
    autotext.set_position((x, y))
    autotext.set_color('white')
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')
    autotext.set_horizontalalignment('center')
    autotext.set_verticalalignment('center')
plt.title("Type of Loans", fontsize=18, fontweight='bold')
legend = plt.legend( wedges, loan_counts.index, title="Purpose", loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)
legend.get_title().set_fontsize(14)
legend.get_title().set_fontweight('bold')
for text in legend.get_texts():
    text.set_fontsize(13)
if save_plots:
    plt.savefig("Loan Intent Dist mako.png", dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()


# function to transform df to % per row
def transform_pct_row(df_in):
    row_sum = df_in.sum(axis=1)
    out_df = df_in.div(row_sum, axis=0)*100
    return out_df

# 7. Loan intent vs repayment outcomes (stacked/grouped bar)
# stacked bar better
loan_counts_crosstab = pd.crosstab(df_no_outlier['loan_intent'], df_no_outlier['loan_status'])
loan_counts_crosstab_pct = transform_pct_row(loan_counts_crosstab)
#loan_counts_crosstab_pct.plot(kind='bar', stacked=True, figsize=(10,6), color=sns.color_palette("tab10", loan_counts_crosstab_pct.shape[1]))
loan_counts_crosstab_pct.plot(kind='bar', stacked=True, figsize=(10,6), color=["darkslateblue", "firebrick"])
plt.title("Loan Intent vs Default Risk", fontsize=18, fontweight='bold')
plt.xlabel("Loan Intent", fontsize=14)
plt.ylabel("%age of Borrowers", fontsize=14)
plt.xticks(rotation=45, ha='center', va='top')
plt.legend(title="Delinquency", fontsize=14, bbox_to_anchor=(1.0, -0.15), loc="upper right")
if save_plots:
    plt.savefig("Loan Intent vs repayment outcomes.png", dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()

# 8. Person with historical default
# stacked better
default_on_file_crosstab = pd.crosstab(df_no_outlier['cb_person_default_on_file'],df_no_outlier['loan_status'])
default_on_file_crosstab_pct = transform_pct_row(default_on_file_crosstab)
#default_on_file_crosstab_pct.plot(kind='bar', stacked=True, figsize=(10,6), color=sns.color_palette("PuBuGn", default_on_file_crosstab_pct.shape[1]))
default_on_file_crosstab_pct.plot(kind='bar', stacked=True, figsize=(10,6), color=["darkslateblue", "firebrick"])
plt.title("Default History vs Default Risk", fontsize=18, fontweight='bold')
plt.xlabel("Default History", fontsize=14)
plt.ylabel("%age of Borrowers", fontsize=14)
plt.xticks(rotation=0, ha='center', va='top',fontsize=14, fontweight='bold')
plt.legend(title="Delinquency")
if save_plots:
    plt.savefig("Default Hist vs Default Risk.png", dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()


# Encode Categorical variables
cat_cols = df_no_outlier.select_dtypes(include=['object','category']).columns
print("Categorical variables:", cat_cols.tolist())
encoded_df = pd.get_dummies(df_no_outlier, columns=cat_cols, drop_first=True)
# Convert bools to integers (0/1)
encoded_df = encoded_df.astype({col: 'int' for col in encoded_df.select_dtypes(include='bool').columns})


x_var = encoded_df.drop(columns='loan_status')
target_var = encoded_df['loan_status']

# Multicollinearity check
x_const = add_constant(x_var)
vif_data = pd.DataFrame()
vif_data["feature"] = x_const.columns[1:] # skip constant
vif_data["VIF"] = [variance_inflation_factor(x_const.values, i)
                   for i in range(1,x_const.shape[1])]
print(vif_data)

# Compute correlation matrix
cols_with_high_vif = ["person_age", "loan_int_rate", "loan_amnt", "cb_person_cred_hist_length"]
for col in encoded_df.columns:
    if col.startswith('loan_grade_'):
        cols_with_high_vif.append(col)
subset_df = encoded_df[cols_with_high_vif]
corr_matrix = subset_df.corr()



# Plot Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.xlabel("Features", fontsize=12, fontweight='bold')
plt.ylabel("Features", fontsize=12, fontweight='bold')
plt.title("Correlation Heatmap for Variables with VIF > 4", fontsize=15, fontweight='bold')
if show_plots:
    plt.show()
if save_plots:
    plt.savefig("correlation_heatmap VIF above 4.png", dpi=300, bbox_inches='tight')
# the main multicollinearity issue is between loan_int_rate and loan_grade. Need to drop one of them or use regularization.

# Heatmap for loan_int_rate and loan_grade
# a = ["loan_int_rate"]
# for col in encoded_df.columns:
#     if col.startswith('loan_grade_'):
#         a.append(col)
# b = encoded_df[a]
# heatmap = b.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(heatmap, annot=True, cmap="twilight", fmt=".1f", annot_kws={"size": 16})
# #plt.xlabel("Features", fontsize=12, fontweight='bold')
# #plt.ylabel("Features", fontsize=12, fontweight='bold')
# plt.title("Correlation Heatmap", fontsize=15, fontweight='bold')
# if save_plots:
#   plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
# if show_plots:
#     plt.show()


# Drop loan_grade dummies as it has very high vif and strong correlation with interest rate
loan_grade_cols = [col for col in x_var.columns if col.startswith('loan_grade_')]
encoded_df_new = x_var.drop(columns=loan_grade_cols)

# Re-run VIF
# Trial
x_const1 = add_constant(encoded_df_new)
vif_data1 = pd.DataFrame()
vif_data1["feature"] = x_const1.columns[1:] # skip constant
vif_data1["VIF"] = [variance_inflation_factor(x_const1.values, i)
                   for i in range(1,x_const1.shape[1])]
print(vif_data1)


### Step 1: fit model on whole data to check p-values and decide on variables
### Step 2: remove unnecessary variables and fit after splitting train and test

# # Hypothesis Testing for Variable Significance
# Fit with statsmodels (for p-values)

# Scaling
scaler1 = StandardScaler()
x_var_scaled = scaler1.fit_transform(encoded_df_new)

import statsmodels.api as sm
const = sm.add_constant(x_var_scaled)
logit_model = sm.Logit(target_var, const)
result = logit_model.fit()
print(result.summary())
print("P-values:\n", result.pvalues)
pval_df = pd.DataFrame({
    'Variable': ["const"]+list(encoded_df_new.columns),
    'p_value': result.pvalues.values
})
print(pval_df)
#pval_df.to_excel("pval_df.xlsx", index=False)


# Remove insignificant variables (with high p-values)
X = encoded_df_new.drop(['person_age','cb_person_cred_hist_length'], axis=1) # data after removing loan_grades, age, credit hist length
Y = target_var


##### Model Fitting
## Model A: Removing person_age and cb_person_cred_hist_length
# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y)
# random_state ensures reproducibility of the split
# stratify=y ensures both sets have a similar proportion of target classes

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fitting model using scikit learn
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train_scaled, Y_train)
y_pred = log_reg_model.predict(X_test_scaled)

y_pred_proba = log_reg_model.predict_proba(X_test_scaled)[:, 1] # Get probabilities for the positive class

# Evaluate the model on the test set
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
accuracy = accuracy_score(Y_test, y_pred) # Y_test are actual values, y_pred are predicted values
print(f"Test Accuracy: {accuracy:.2f}")
cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(cm)
cr = classification_report(Y_test, y_pred)
print(cr)
auc = roc_auc_score(Y_test, y_pred_proba)
print(f"AUC: {auc:.3f}\n")


# Confusion Matrix heatmap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.patches import Rectangle
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Set3_r', cbar=False, linewidths=1, linecolor='white', annot_kws={'size': 14, 'weight': 'bold'})
#ax = plt.gca()
#ax.add_patch(plt.Rectangle((0-0.5, 1-0.5), 1, 1, fill=True, color='#ff6666',alpha=0.55, zorder=10, linewidth=1.8))
plt.title('Confusion Matrix - Loan Default Prediction', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
tick_labels = ['Non-Default (0)', 'Default (1)']
plt.xticks(ticks=[0.5, 1.5], labels=tick_labels, fontsize=11)
plt.yticks(ticks=[0.5, 1.5], labels=tick_labels, fontsize=11, rotation=0)

plt.tight_layout()
if show_plots:
    plt.show()
if save_plots:
    plt.savefig('Confusion Matrix - Loan Default Prediction.png', dpi=300)




# ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity (TPR)')
plt.title('ROC Curve - Logistic Regression', fontweight='bold')
plt.legend(loc='lower right')
if save_plots:
    plt.savefig('ROC Curve.png', dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(Y_test, y_pred_proba)
avg_precision = average_precision_score(Y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='blue', label=f'PR Curve (AP = {avg_precision:.2f})')
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision-Recall Curve for Logistic Regression", fontweight='bold')
plt.legend(loc="lower left", fontsize=12)
if save_plots:
    plt.savefig('PR Curve.png', dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()


## ______________________

## Model B: Including age, credit history length
X_train_B, X_test_B, Y_train_B, Y_test_B = train_test_split(
    encoded_df_new, Y, test_size=0.3, random_state=42, stratify=Y)
scaler_B = StandardScaler()
X_train_B_scaled = scaler_B.fit_transform(X_train_B)
X_test_B_scaled = scaler_B.transform(X_test_B)
log_reg_model_B = LogisticRegression(max_iter=1000)
log_reg_model_B.fit(X_train_B_scaled, Y_train_B)
predictions_B = log_reg_model_B.predict(X_test_B_scaled)
pred_prob_B = log_reg_model_B.predict_proba(X_test_B_scaled)[:, 1]

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
accuracyB = accuracy_score(Y_test, predictions_B) # Y_test are actual values, y_pred are predicted values
print(f"Test Accuracy of Model B: {accuracyB:.2f}")
cmB = confusion_matrix(Y_test_B, predictions_B)
print("Confusion Matrix:")
print(cmB)
crB = classification_report(Y_test_B, predictions_B)
print(crB)
aucB = roc_auc_score(Y_test_B, predictions_B)
print(f"AUC: {aucB:.3f}\n")

## __________________________________________________
# ***** Based on AUC-ROC Model A has been selected as final model.*******


# Calibration Plot (Reliability Diagram)
# from sklearn.calibration import calibration_curve
# from sklearn.metrics import brier_score_loss
# prob_true, prob_pred = calibration_curve(
#     Y_test,
#     y_pred_proba,
#     n_bins=10,              # or 15, try what looks clean
#     strategy='uniform')    # or quantile
# brier = brier_score_loss(Y_test, y_pred_proba)
# plt.figure(figsize=(10, 8))
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
# plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='blue', label=f'My model (Brier = {brier:.2f})')
# plt.xlabel('Predicted probability of default', fontsize=14)
# plt.ylabel('Actual fraction of defaults', fontsize=14)
# plt.title('Calibration Plot / Reliability Diagram', fontsize=18, fontweight='bold')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(loc='lower right')
# plt.grid(True, alpha=0.3)
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# if show_plots:
#     plt.show()


# Confusion matrix at difft thresholds based on Model A
def confusion_matrices_at_thresholds(
    Y_test,
    y_pred_proba,
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    normalize = False):
    """
    Prints confusion matrices for multiple probability thresholds.
    Y_test: true binary labels (0/1)
    y_pred: predicted probabilities of class 1 (default)
    """
    results = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred, labels=[0, 1]).ravel()

        row = {
            'Threshold': f"{thresh:.2f}",
            'TP': tp,
            'FN': fn,
            'FP': fp,
            'TN': tn,
            'Recall (TPR)': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn)
         }
        results.append(row)
    cols_order = ['Threshold', 'TP', 'FN', 'FP', 'TN',
                  'Recall (TPR)', 'Precision', 'FPR', 'Accuracy']
    df = pd.DataFrame(results)[cols_order]
    return df
confusion_matrices = confusion_matrices_at_thresholds(Y_test,y_pred_proba)
print(confusion_matrices)
#confusion_matrices.to_excel("confusion_matrices.xlsx", index=False)

# Coefficient Analysis
# Coefficients/Odds Ratio
# Get the p-values using statsmodels for Model A
const1 = sm.add_constant(X_train_scaled)
logit_model1 = sm.Logit(Y_train, const1)
fit = logit_model1.fit()
print("P-values:\n", fit.pvalues)
p_values = fit.pvalues.values

coefficients = log_reg_model.coef_[0]
abs_coef = np.abs(coefficients)
feature_names = X_train.columns
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': abs_coef,
    'Odds Ratio': np.exp(coefficients)
})
coef_df['p-value'] = p_values[1:].round(4)
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
def significance_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''
coef_df['Signif.'] = coef_df['p-value'].apply(significance_stars)
styled = coef_df.style.format({
    'Coefficient': '{:.4f}',
    'Odds Ratio' : '{:.3f}'}).background_gradient(subset=['Odds Ratio'], cmap='RdYlGn_r').set_caption("Logistic Regression Coefficients, Odds Ratios & Significance")  # red = risk ↑, green = protective

#coef_df.to_excel("model_coefficients.xlsx", index=False)

## Bar chart of Odds Ratio
plot_data = coef_df.sort_values('Odds Ratio', ascending=False).copy()
colors = ["#0000FF" if x > 1 else "#FFBF00" for x in plot_data['Odds Ratio']]
#colors = ["#734F96" if x > 1 else "#FFFF99" for x in plot_data['Odds Ratio']] # for ppt
plt.figure(figsize=(10, 6))
sns.barplot(x='Odds Ratio', y='Feature', data=plot_data, hue='Feature', palette=colors, dodge=False, legend=False, edgecolor='black')
plt.axvline(x=1, color='#E30022', linestyle='--', linewidth=1.3, label='Neutral (OR = 1)')
plt.xlabel('Odds Ratio (exp(β))', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Odds Ratios of Key Predictors – Loan Default Model\n(> 1 = increases odds of default)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=12)
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
if save_plots:
    plt.savefig('Bar Chart of Odds Ratio.png',  dpi=300, bbox_inches='tight')
if show_plots:
    plt.show()







# # Connecting this trained model to Streamlit app
# if streamlit_dump:
#     import joblib
#     # joblib.dump(log_reg_model, "log_reg_model.pkl")
#     joblib.dump(scaler,"scaler.pkl")
#     joblib.dump({
#         'model': log_reg_model,
#         'feature_names': X_train.columns.tolist()}, "logistic_model_with_feature_names.pkl")
#     joblib.dump(df_no_outlier, "df_no_outlier.pkl")
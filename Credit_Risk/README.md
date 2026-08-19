# Loan Default Risk Prediction Using Logistic Regression

Predicting a borrower's probability of loan default from application-level data, built end to end in Python and deployed as an interactive Streamlit app.

## Overview

Lenders need to assess a borrower's risk of default quickly and consistently, at a volume manual review simply can't handle. This project builds a logistic regression model that predicts the probability of loan default from 12 borrower and loan attributes, then deploys it as a live Streamlit tool that returns a risk score in real time. The dataset covers over 32,000 loan applications, and the final model reaches an AUC of 0.85, with results validated on a held-out 30% test set.

## The Problem

Manual credit assessment doesn't scale, and inconsistent judgment calls between reviewers create risk of their own. A statistical model that scores default risk consistently, using signals like income, loan amount, and credit history, gives a lender a faster and more defensible first pass, flagging genuinely risky applications for closer review rather than treating every application the same way.

## Data

The dataset is the Kaggle "Credit Risk Dataset," containing 32,581 records and 12 features covering borrower demographics (age, income, home ownership, employment length), loan details (amount, intent, grade, interest rate), and credit history (prior default on file, credit history length). The target variable, loan_status, flags whether the loan defaulted.

Two data quality issues needed addressing before modeling. loan_int_rate was missing for a portion of records, person_emp_length was missing for a smaller portion. Both are documented and handled in the cleaning step below.

## My Approach

**Missing data:** Rather than dropping every row with a missing interest rate, I checked whether it correlated with another field first. It turned out to correlate strongly with loan_grade, so missing rates were imputed using the median rate for that borrower's grade rather than an overall average, which preserves the real relationship between grade and pricing instead of flattening it. Missing values in person_emp_length were a small enough share of the data that those rows were dropped rather than imputed.

**Outlier handling:** Numeric fields were checked with box plots, then cleaned using a z-score threshold, removing records that sit too many standard deviations away from the average for any given variable. This is gentler than a hard percentile cutoff and keeps genuinely unusual but valid records intact.

**Exploratory Data Analysis:** Before modeling, I looked at how default rates varied across age, income, home ownership, loan intent, interest rate bands, and prior default history, to build intuition for which features were likely to matter before letting statistics confirm or deny it.

**Encoding and multicollinearity:** Categorical variables were one-hot encoded. A Variance Inflation Factor (VIF) check then revealed strong multicollinearity between loan_grade and loan_int_rate, which makes sense since grade largely determines the interest rate a borrower is offered. Rather than keeping both and letting the model split credit between two versions of the same signal, the loan grade dummy variables were dropped in favor of keeping interest rate, which carries the same information in a single continuous field.

**Feature selection:** A logistic regression was first fit using statsmodels purely to inspect p-values. Two features, borrower age and credit history length, came back statistically insignificant and were dropped from the final feature set.

**Model comparison:** Two versions of the model were trained and compared, one excluding age and credit history length (Model A) and one including them (Model B). Model A was selected as the final model based on a stronger AUC-ROC score, confirming that the earlier feature selection step genuinely improved the model rather than just simplifying it.

**Final model:** Data was split 70/30 into train and test sets using stratified sampling, so both sets preserve the same proportion of defaults as the full dataset. Features were standardized, and a logistic regression was fit on the training set. Logistic regression was chosen deliberately over more complex models like random forest or gradient boosting, because credit decisions need to be explainable to both regulators and applicants, and logistic regression's coefficients translate directly into odds ratios that are straightforward to interpret and justify.

## Key Insights

Loan-to-income ratio was among the strongest predictors of default, borrowers committing a higher share of income to loan repayment defaulted at a meaningfully higher rate.
Prior default on file was associated with a sharply higher odds ratio for default, unsurprising, but the model quantifies exactly how much higher.

<img width="300" alt="image" src="https://github.com/user-attachments/assets/df826e56-8b13-4c1d-bdf6-d58624253b77" />


<img width="400"  alt="image" src="https://github.com/user-attachments/assets/5f77a992-bf24-45e5-afd5-f7eda9394640" />




**Threshold strategy** Rather than relying on a single fixed cutoff, the model was evaluated across a range of probability thresholds, since the right cutoff genuinely depends on how much risk a lender is willing to carry.

- **Conservative Strategy** threshold (around 0.25 to 0.30) flags more borrowers as high risk, catching more potential defaults at the cost of some false alarms. This suits a lender who already has high loan exposure and wants to keep risk appetite low.
- **Aggressive Strategy** threshold (around 0.50) flags fewer borrowers, approving more loans while accepting a bit more default risk. This suits a lender pursuing aggressive loan book growth, where controlled risk tolerance is an acceptable trade-off for volume.

## Practical impact
This threshold flexibility is what makes the model genuinely usable in underwriting, rather than just an academic classifier. A risk team doesn't have to accept a single one-size-fits-all cutoff, they can dial the threshold up or down depending on current lending strategy, portfolio exposure, or economic conditions, without retraining the model itself. For a lender, that translates directly into practical decisions: which threshold to set for automatic approval versus manual review, how much of the applicant pool to greenlight in a given quarter, and how to balance growth targets against acceptable loss rates, all from the same underlying model.

## Limitations and Future Work

Logistic regression assumes a linear relationship between features and the log-odds of default, which may miss non-linear interactions a tree-based model could capture, at the cost of the interpretability that makes this model suitable for a regulated lending context in the first place. A useful next step would be benchmarking against a random forest or gradient boosting model to quantify that trade-off directly, alongside adding macroeconomic variables, which real credit risk models almost always include, to test how the model holds up outside a single static dataset.

## Tech Stack

Python, pandas, scikit-learn, statsmodels, seaborn/matplotlib, Streamlit

## Acknowledgments

Dataset sourced from Kaggle.

## License

MIT


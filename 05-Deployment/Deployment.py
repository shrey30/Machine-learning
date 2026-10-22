#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/refs/heads/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'


# In[3]:


#!wget $data


# In[4]:


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[5]:


df.head()


# In[6]:


df.head().T


# # Data Preparation

# In[7]:


df.dtypes


# In[8]:


# We can see that TotalCharges is a str but it should be an int. So we will convert it.


# In[9]:


pd.to_numeric(df.TotalCharges, errors='coerce')
#In pd.to_numeric(), coerce means:
#If a value cannot be converted to a number, replace it with NaN instead of throwing an error.


# In[10]:


df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')


# In[11]:


df.columns = df.columns.str.lower().str.replace(' ', '_')


# In[12]:


df.columns


# In[13]:


df.dtypes


# In[14]:


categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)


# In[15]:


for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# In[16]:


df.head().T


# In[17]:


df.churn


# In[18]:


df.churn = (df.churn == 'yes').astype(int)


# In[19]:


df.churn.head()


# In[20]:


df.isnull().sum()


# In[21]:


df.totalcharges = df.totalcharges.fillna(0)


# # Setting Up Validation Framework

# * ### Splitting this dataset into training/validation/test using Scikit-Learn

# In[22]:


from sklearn.model_selection import train_test_split


# #### train_test_split is a function that breaks the dataset into two parts, training and test. But we will split it more for validation.

# In[23]:


df_full_train, df_test = train_test_split(df, test_size= 0.2, random_state= 1)


# In[24]:


len(df_full_train), len(df_test)


# #### Now our data is split into 80:20, but from that 80 percent we need to make it 60:20. 20% of 80% is 25%.

# In[25]:


df_train, df_val = train_test_split(df_full_train, test_size= 0.25, random_state= 1)


# In[26]:


len(df_train), len(df_val), len(df_test)


# In[27]:


df_train.head()


# In[28]:


df_full_train = df_full_train.reset_index(drop= True)
df_train = df_train.reset_index(drop= True)
df_val = df_val.reset_index(drop= True)
df_test = df_test.reset_index(drop= True)


# In[29]:


df_train.head()


# In[30]:


y_full_train = df_full_train['churn'].values
y_train = df_train['churn'].values
y_val = df_val['churn'].values
y_test = df_test['churn'].values


# In[31]:


del df_train['churn']
del df_val['churn']
del df_test['churn']


# In[32]:


df_train.head().T


# # Exploratory Data Analysis

# * #### Check missing values.
# * #### Look at target value(churn).
# * #### Look at numerical and categorical variables.

# In[33]:


df.isnull().sum()


# In[34]:


np.unique(y_full_train, return_counts= True)
# Basically 0 = 73%, 1 = 27%


# In[35]:


global_churn_rate = y_full_train.mean()
round(global_churn_rate, 2)


# In[36]:


df_full_train.dtypes


# In[37]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']


# In[38]:


df_full_train.dtypes.index


# In[39]:


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
          'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# In[40]:


categorical


# In[41]:


df[categorical].nunique()


# In[ ]:





# # Feature importance: Churn Rate and Risk Ratio

# #### Feature importance analysis(part of EDA) - Identifying which features affect our target variable.

# * #### Churn rate
# * #### Risk ratio
# * #### Mutual information

# #### Churn rate and Risk ratio

# #### Refer notebook for notes on these concepts.

# In[42]:


global_churn_rate = df_full_train.churn.mean()
global_churn_rate


# In[43]:


churn_female = df_full_train[df_full_train['gender'] == 'female'].churn.mean()
churn_female


# In[44]:


churn_male = df_full_train[df_full_train['gender'] == 'male'].churn.mean()
churn_male


# In[45]:


churn_partner = df_full_train[df_full_train['partner'] == 'yes'].churn.mean()
churn_partner


# In[46]:


churn_no_partner = df_full_train[df_full_train['partner'] == 'no'].churn.mean()
churn_no_partner


# In[ ]:





# In[47]:


df_full_train.groupby('gender').churn.mean()


# In[48]:


df_full_train.groupby('gender').churn.agg(['mean', 'count'])
# This creates a dataframe for us that divides the data into caregories, and apply functions on it.


# In[49]:


df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count'])
df_group['Diff'] = global_churn_rate - df_group['mean']
df_group['Risk'] = df_group['mean'] / global_churn_rate

df_group


# #### Let's do this for all categorical variables.

# In[50]:


from IPython.display import display


# In[51]:


for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['Diff'] = global_churn_rate - df_group['mean']
    df_group['Risk'] = df_group['mean'] / global_churn_rate
    print(c)
    display(df_group)
    print()
    print()


# ### Mutual Information

# #### Go to this link for proper explanation. - https://www.youtube.com/watch?v=eJIp_mgVLwE&t=379s

# In[52]:


from sklearn.metrics import mutual_info_score


# In[53]:


mutual_info_score(df_full_train.churn, df_full_train.gender)


# #### Similarly we can apply this function to all the categoical columns.

# In[54]:


def mutual_info_churn_score(column):
    return mutual_info_score(column, df_full_train.churn)


# In[55]:


mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending = False)


# #### Now we can see the mutual information of each categorical variable with churn.

# ### Correlation

# #### For further concepts, read this concepts for better understanding.

# #### Mean, Variance and Standard Deviation - https://www.youtube.com/watch?v=SzZ6GpcfoQY

# #### Covariance - https://youtu.be/qtaqvPAeEJY

# #### Correlation - https://youtu.be/xZ_z8KWkhXE

# In[56]:


df_full_train[numerical]


# In[57]:


df_full_train[numerical].corrwith(df_full_train.churn)


# # One-hot Encoding

# #### Using Scikit-Learn to encode categorical features.

# In[58]:


from sklearn.feature_extraction import DictVectorizer


# #### Dictvectorizer uses dictionaries as input to encode it.
# #### Let us see an example.

# In[59]:


df_train[['gender', 'contract']].iloc[:100].to_dict(orient='records')
# If we don't specify orient, it will make dictionaries column wise.


# In[60]:


dicts = df_train[['gender', 'contract']].iloc[:100].to_dict(orient='records')


# In[61]:


dv = DictVectorizer(sparse = False)
# We don't want it to make sparse matrix of the encoded values.


# In[62]:


dv.fit(dicts)
# It learns about the features.


# In[63]:


dv.feature_names_


# In[64]:


dv.transform(dicts)
# It actually works on the data here.


# #### DictVectorizer is actually pretty smart, if you give it numerical variable, it will not encode it.

# In[65]:


dicts = df_train[['gender', 'contract', 'tenure']].iloc[:100].to_dict(orient='records')


# In[66]:


dv.fit(dicts)


# In[67]:


dv.transform(dicts)


# #### Now, let's work on our training data.

# In[68]:


train_dicts = df_train[categorical + numerical].to_dict(orient= 'records')


# #### We can do fit and transform into a single function called fit_transform.

# In[69]:


X_train = dv.fit_transform(train_dicts)
X_train


# In[70]:


list(X_train[0])


# #### Similarly, we'll do encoding of validation data.

# In[71]:


val_dicts = df_val[categorical + numerical].to_dict(orient= 'records')


# In[72]:


X_val = dv.transform(val_dicts)
# We don't need to fit it again, because it knows about the variables.
X_val


# # Logictic Regression

# #### https://youtu.be/yIYKR4sgzI8

# #### https://youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe&si=Ak5EHd2EJeF0znMN

# #### https://www.youtube.com/watch?v=3bvM3NyMiE0

# #### https://www.youtube.com/watch?v=EKm0spFxFG4

# #### Instead of the summation in linear regression, the logistic regression uses Sigmoid function in the formula.

# #### Let's use Scikit Learn to implement Logistic Regression.

# In[73]:


from sklearn.linear_model import LogisticRegression


# In[74]:


model = LogisticRegression()


# In[75]:


# Let's train the model
model.fit(X_train, y_train)


# In[76]:


# It's the bias term.
model.intercept_[0]


# In[77]:


model.coef_
# These are the weights given by the model.


# In[78]:


np.column_stack((model.predict(X_train)[:10],model.predict_proba(X_train)[:10]))
# predict_proba has two columns, it has soft predictions, it has scores, probability of being 0 or churn and probability of being 1 or churn.


# #### Since we have trained our model on training data, let's see how it performs against validation data.

# In[79]:


y_pred = model.predict_proba(X_val)[:,1]
y_pred


# In[80]:


churn_decision = (y_pred >= 0.5)
churn_decision


# In[81]:


# We also have y_val from our data.
(churn_decision == y_val).mean()
# It means that our model is 80% correct.


# In[82]:


df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype('int')
df_pred['actual'] = y_val
df_pred['correct'] = df_pred['prediction'] == df_pred['actual']

df_pred


# In[83]:


# Here we can see that when we use mean() function on boolean data, it automatically converts it to int and then finds mean.
df_pred.correct.mean()


# # Model Interpretation

# #### https://youtu.be/OUrlxnUAAEA

# # Using the Model

# #### Let's train the model with full data and then test it against the testing data.

# In[84]:


dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')


# #### Now, let's encode the data.

# In[85]:


dv = DictVectorizer(sparse = False)


# In[86]:


X_full_train = dv.fit_transform(dicts_full_train)


# In[87]:


X_full_train


# In[88]:


dv.feature_names_


# In[89]:


model = LogisticRegression()


# In[90]:


model.fit(X_full_train, y_full_train)


# In[ ]:





# In[ ]:





# # Accuracy and Dummy Model

# * #### Evaluate the model on different thresholds.

# * #### Check the accuracy of dummy baselines.

# In[91]:


y_pred = model.predict_proba(X_val)[:,1]


# In[92]:


churn_decision = (y_pred >= 0.5)


# In[93]:


(churn_decision == y_val).mean()


# #### Here we have 80% accuracy with 0.5 as our threshold on validation data.

# #### Let's see how it performs with different threshold values.

# #### For doing the same as we did above, we will use Scikit-Learn.

# In[94]:


from sklearn.metrics import accuracy_score


# In[95]:


thresholds = np.linspace(0, 1, 21)


# In[96]:


thresholds


# In[97]:


scores = []
for t in thresholds:
    score = accuracy_score(y_val, (y_pred >= t))
    scores.append(score)
    print(f'{t:.2f} {score:.3f}')


# In[98]:


sns.lineplot(x=thresholds, y=scores)

plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.show()


# #### The best threshold is 0.55. We can see a peak in the graph too.

# #### We can also see that after 0.9, the accuracy remains same. This means that if we considered all of the customers as not churning, we would be only 8% wrong

# #### If we see the validation data, only 27% people had churned. Out of that 27% our model works right 80% of the time

# In[99]:


from collections import Counter


# In[100]:


Counter(y_val)


# #### The ratio of people churing is 1:3

# In[101]:


round(y_val.mean(),2)


# #### Therefore, if we consider that everyone is churning,i.e. threshold = 1. Our dummy model is 73% accurate. And our actual model is 80% accurate with a threshold of 0.5.

# # Confusion Matrix

# #### https://www.youtube.com/watch?v=Kdsp6soqA7o&t=1s

# In[102]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[103]:


t= 0.55 # Why 0.55? Because we saw that our model was more accurate at this threshold value.
predicted_positive = (y_pred >= t)
predicted_negative = (y_pred < t)


# In[104]:


# Now we will use bitwise operator to find the values of confusion matrix.
true_positives = (actual_positive & predicted_positive).sum()
true_negatives = (actual_negative & predicted_negative).sum()
false_positives = (actual_negative & predicted_positive).sum()
false_negatives = (actual_positive & predicted_negative).sum()


# In[105]:


print(f'tp:{true_positives} \ntn:{true_negatives}\nfp:{false_positives}\nfn:{false_negatives}')


# In[106]:


confusion_matrix = np.array([
    [true_positives, false_positives],
    [false_negatives, true_negatives]
])
confusion_matrix


# In[107]:


(confusion_matrix/confusion_matrix.sum()).round(2)


# #### We know that our model is 81% percent accurate.
# #### Through confusion matrix, we got a clear picture.
# #### **Accuracy(81%) = tp(14%) + tn(67%)**

# # Precision and Recall

# #### https://youtu.be/qWfzIYCvBqo

# In[108]:


precision = true_positives / (true_positives + false_positives)
precision
# precision = total_apples_correct / total_apple_side_observations.


# In[109]:


true_positives


# In[110]:


true_positives + false_positives


# #### Therefore our precision is 73% for churn prediction.

# In[111]:


recall = true_positives / (true_positives + false_negatives)
recall
# Recall is total_apples_correct / total_actual_apples


# #### And our recall is 51%

# #### Remember this:
# #### First column of confusion matrix can help us calculate recall.
# #### First row of confusion matrix can help us calculate precision.

# # ROC Curves
# ## Receiver Operating Characteristic (ROC)

# #### https://youtu.be/4jRBRDbJemM

# #### True positvity rate (Sensitivity) and recall are same metrics.
# #### False positive rate = 1 - Specificity = Can be calculated from second column of the confusion matrix.

# #### Btw specificity is also known as True negativity rate. 
# #### True negativity rate = TN / TN + FP

# #### ROC graph has True positivity rate on y-ais and False positivity rate on x-axis.

# #### The ROC graph summarizes all the confusion matrics that each threshold produced.

# In[112]:


# Let's first calculate TRP and FRP of our data.


# In[113]:


tpr = true_positives / (true_positives + false_negatives)
tpr


# In[114]:


fpr = false_positives / (false_positives + true_negatives)
fpr


# In[115]:


# Now, let's calculate TPR and FPR for different threshold values.

thresholds = np.linspace(0, 1, 101)
scores = []

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predicted_positive = (y_pred >= t)
    predicted_negative = (y_pred < t)


    true_positives = (actual_positive & predicted_positive).sum()
    true_negatives = (actual_negative & predicted_negative).sum()
    false_positives = (actual_negative & predicted_positive).sum()
    false_negatives = (actual_positive & predicted_negative).sum()

    scores.append((t, true_positives, false_positives, false_negatives, true_negatives))


# In[116]:


scores


# In[117]:


df_scores = pd.DataFrame(scores)
columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores.columns = columns


# In[118]:


df_scores[::10]


# In[119]:


df_scores['TPR'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['FPR'] = df_scores.fp / (df_scores.fp + df_scores.tn)


# In[120]:


df_scores[::10]


# In[155]:


sns.lineplot(
    x=df_scores.FPR,
    y=df_scores.TPR,
    linewidth=2.5
)

plt.plot([0, 1], [0, 1], '--', linewidth=1.5)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.grid(alpha=0.25)
plt.xlim(0, 1)
plt.ylim(0, 1.1)

plt.show()


# In[156]:


sns.lineplot(x=df_scores.threshold, y=df_scores.TPR, label="TPR")
sns.lineplot(x=df_scores.threshold, y=df_scores.FPR, label="FPR")

plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("TPR and FPR vs Threshold")

plt.grid(alpha=0.25)
plt.legend()

plt.show()


# #### We prefer FPR curve as steep as possible.

# ### Random model

# In[157]:


np.random.seed(1)
y_rand = np.random.uniform(0, 1, size= len(y_val)).round(2)


# In[158]:


y_rand


# #### Let's see the TPR and FPR for random model.

# In[159]:


def tpr_fpr_dataframe(y_val, y_pred):
    thresholds = np.linspace(0, 1, 101)
    scores = []

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predicted_positive = (y_pred >= t)
        predicted_negative = (y_pred < t)


        true_positives = (actual_positive & predicted_positive).sum()
        true_negatives = (actual_negative & predicted_negative).sum()
        false_positives = (actual_negative & predicted_positive).sum()
        false_negatives = (actual_positive & predicted_negative).sum()

        scores.append((t, true_positives, false_positives, false_negatives, true_negatives))

        df_scores = pd.DataFrame(scores)
        columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
        df_scores.columns = columns

        df_scores['TPR'] = df_scores.tp / (df_scores.tp + df_scores.fn)
        df_scores['FPR'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores


# In[160]:


df_rand = tpr_fpr_dataframe(y_val, y_rand)


# In[161]:


df_rand[::10]


# In[162]:


sns.lineplot(x=df_rand.threshold, y=df_rand.TPR, label="TPR")
sns.lineplot(x=df_rand.threshold, y=df_rand.FPR, label="FPR")

plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("TPR and FPR vs Threshold of Random Model")

plt.grid(alpha=0.25)
plt.legend()

plt.show()


# ### Ideal Model

# #### An ideal model would have 100% accuracy.
# #### for that we need to arrange the customers in acending order from no churn to churn.

# In[163]:


y_val


# In[164]:


np.unique(y_val)


# In[165]:


num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
num_neg, num_pos


# In[166]:


y_ideal = np.repeat([0,1], [num_neg, num_pos])


# In[167]:


y_ideal


# In[168]:


y_ideal_pred = np.linspace(0, 1, len(y_val))


# In[169]:


1 - y_val.mean()


# In[170]:


((y_ideal_pred >= 0.726) == y_ideal).mean()


# In[171]:


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
df_ideal[::10]


# In[172]:


sns.lineplot(x=df_ideal.threshold, y=df_ideal.TPR, label="TPR")
sns.lineplot(x=df_ideal.threshold, y=df_ideal.FPR, label="FPR")

plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("TPR and FPR vs Threshold of Ideal Model")

plt.grid(alpha=0.25)
plt.legend()

plt.show()


# #### Let's look at the ROC curve of our model again using sklearn.

# In[173]:


from sklearn.metrics import roc_curve


# In[174]:


fpr, tpr ,thresholds = roc_curve(y_val, y_pred)


# In[175]:


sns.lineplot(
    x=fpr,
    y=tpr,
    linewidth=2.5
)

plt.plot([0, 1], [0, 1], '--', linewidth=1.5)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.grid(alpha=0.25)
plt.xlim(0, 1)
plt.ylim(0, 1.1)

plt.show()


# # ROC Area Under The Curve (AUC)

# In[176]:


from sklearn.metrics import auc
# auc finds area under any curve


# In[177]:


fpr, tpr ,thresholds = roc_curve(y_val, y_pred)


# In[178]:


auc(fpr, tpr)


# In[179]:


from sklearn.metrics import roc_auc_score
# This finds auc of roc directly through values.


# In[180]:


auc = roc_auc_score(y_val, y_pred)
auc


# In[181]:


# This visualization is from GPT.

plt.figure(figsize=(9, 7))

# ROC curve
sns.lineplot(
    x=fpr,
    y=tpr,
    linewidth=3.5,
    color="#7C3AED",
    label=f"ROC Curve  •  AUC = {auc:.3f}"
)

# Shaded AUC region
plt.fill_between(
    fpr,
    tpr,
    color="#A78BFA",
    alpha=0.30
)

# Random classifier
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=2,
    color="#94A3B8",
    label="Random Classifier"
)

# Make the plot feel cleaner
plt.xlabel("False Positive Rate", fontsize=13, fontweight="medium")
plt.ylabel("True Positive Rate", fontsize=13, fontweight="medium")

plt.title(
    "Receiver Operating Characteristic",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlim(0, 1)
plt.ylim(0, 1.1)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.2
)

plt.legend(
    loc="lower right",
    frameon=True,
    fancybox=True,
    framealpha=0.9
)

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# # Cross Validation

# #### Explanation - https://youtu.be/fSytzGwwBVw

# * #### Evaluating the same model on different subset of data.
# * #### Getting the average prediction and spread of prediction.

# In[182]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[numerical + categorical].to_dict(orient= 'records')
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(max_iter = 5000, C=C)
    model.fit(X_train, y_train)

    return dv, model


# In[183]:


dv, model = train(df_train, y_train)


# In[184]:


def predict(dv, model, df):
    dicts = df[categorical + numerical].to_dict(orient= 'records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[185]:


predict(dv, model, df_val)


# In[186]:


from sklearn.model_selection import KFold


# In[187]:


kfold = KFold(n_splits = 10, shuffle = True, random_state= 1)


# In[190]:


from tqdm.auto import tqdm


# In[191]:


scores= []

for train_idx, val_idx in tqdm(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train)
    y_pred = predict(dv, model, df_val)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'{np.mean(scores)}, {np.std(scores)}')


# In[192]:


# Finally training the model
dv, model = train(df_full_train, df_full_train.churn.values, C=1)
y_pred = predict(dv, model, df_test)

auc = roc_auc_score(y_test, y_pred)
auc


# In[ ]:





# In[ ]:





# # Save the Model

# In[193]:


import pickle


# In[194]:


C = 1
output_file = f'model_C={C}.bin'
output_file


# In[195]:


with open(output_file, 'wb') as f_out: # 'wb' because we are writing into binary file.
    pickle.dump((dv, model), f_out)

# This method automatically closes the file.


# In[ ]:





# # Load the Model

# In[196]:


import pickle


# In[197]:


model_file = 'model_C=1.bin'


# In[198]:


with open(model_file, 'rb') as f_in: # 'rb' because we are reading into binary file.
    dv, model = pickle.load(f_in)


# In[199]:


dv, model


# #### See we have our model back in our notebook and we can use it like we have done it soooo many times.

# In[ ]:





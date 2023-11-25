#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


copper_industry = pd.read_csv("C:\\Users\\Ivin\\Downloads\copper_set.csv")


# In[3]:


copper_industry


# In[4]:


copper_industry.head()


# In[5]:


copper_industry.columns


# In[6]:


copper_industry.info()


# In[7]:


len(copper_industry)


# In[8]:


copper_industry.shape


# In[9]:


print(len(copper_industry["item_date"].unique()))
print(len(copper_industry["customer"].unique()))
print(len(copper_industry["country"].unique()))
print(len(copper_industry["item type"].unique()))
print(len(copper_industry["material_ref"].unique()))
print(len(copper_industry["product_ref"].unique()))
print(len(copper_industry["delivery date"].unique()))


# ###### Handling null (or) na values

# In[10]:


copper_industry.isna().sum()


# In[11]:


copper_industry.isna().sum()/len(copper_industry)*100


# ###### find the numeric and categorical columns

# In[12]:


categorical_cols = [i for i in copper_industry.columns if copper_industry[i].dtype =="object"]


# In[13]:


print(categorical_cols)


# In[14]:


numeric_cols = [ i for i in copper_industry if i not in categorical_cols]


# In[15]:


print(numeric_cols)


# In[16]:


copper_industry ['item_date'] = pd.to_datetime(copper_industry['item_date'], format='%Y%m%d', errors='coerce').dt.date
copper_industry ['quantity tons'] = pd.to_numeric(copper_industry['quantity tons'], errors='coerce')
copper_industry ['customer'] = pd.to_numeric(copper_industry['customer'], errors='coerce')
copper_industry ['country'] = pd.to_numeric(copper_industry['country'], errors='coerce')
copper_industry ['application'] = pd.to_numeric(copper_industry['application'], errors='coerce')
copper_industry ['thickness'] = pd.to_numeric(copper_industry['thickness'], errors='coerce')
copper_industry ['width'] = pd.to_numeric(copper_industry['width'], errors='coerce')
copper_industry ['material_ref'] = copper_industry['material_ref'].str.lstrip('0')
copper_industry ['product_ref'] = pd.to_numeric(copper_industry['product_ref'], errors='coerce')
copper_industry ['delivery date'] = pd.to_datetime(copper_industry['delivery date'], format='%Y%m%d', errors='coerce').dt.date
copper_industry ['selling_price'] = pd.to_numeric(copper_industry['selling_price'], errors='coerce')


# In[17]:


copper_industry.head()


# In[18]:


copper_industry.isna().sum()


# In[19]:


copper_industry.isna().sum()/len(copper_industry)*100


# In[20]:


copper_industry["material_ref"].fillna("unknown",inplace = True)
copper_industry = copper_industry.dropna()


# In[21]:


copper_industry.isna().sum()


# In[22]:


copper_data = copper_industry.copy()


# In[23]:


sns.violinplot(x=copper_data["width"])


# In[24]:


sns.violinplot(x=copper_data["application"])


# In[25]:


sns.violinplot(copper_data["country"])


# In[26]:


sns.boxplot(copper_data)


# In[27]:


sns.distplot(copper_data["selling_price"])
plt.show()

sns.distplot(copper_data["quantity tons"])
plt.show()

sns.distplot(copper_data["country"])
plt.show()

sns.distplot(copper_data["thickness"])
plt.show()


# In[28]:


mask1 = copper_data['selling_price'] <= 0
print(mask1.sum())
copper_data.loc[mask1, 'selling_price'] = np.nan

mask1 = copper_data['quantity tons'] <= 0
print(mask1.sum())
copper_data.loc[mask1, 'quantity tons'] = np.nan

mask1 =copper_data['thickness'] <= 0
print(mask1.sum())


# In[29]:


copper_data.isna().sum()


# In[30]:


copper_data.isna().sum()/len(copper_data)*100


# In[31]:


copper_data['selling_price_log'] = np.log(copper_data['selling_price'])
sns.distplot(copper_data['selling_price_log'])
plt.show()

copper_data['quantity tons_log'] = np.log(copper_data['quantity tons'])
sns.distplot(copper_data['quantity tons_log'])
plt.show()

copper_data['thickness_log'] = np.log(copper_data['thickness'])
sns.distplot(copper_data['thickness_log'])
plt.show()


# In[32]:


copper_data.head()


# In[33]:


x=copper_data[['quantity tons_log','application','thickness_log','width','selling_price_log','country','customer','product_ref']].corr()


# In[34]:


sns.heatmap(x, annot=True, cmap="YlGnBu")


# Encoding

# In[35]:


from sklearn.preprocessing import OrdinalEncoder


# In[36]:


OE = OrdinalEncoder()
copper_data.status = OE.fit_transform(copper_data[['status']])
copper_data['item type'] = OE.fit_transform(copper_data[['item type']])


# In[37]:


copper_data=copper_data.dropna()


# In[38]:


copper_data.isna().sum()/len(copper_data)*100


# In[39]:


x=copper_data[['quantity tons_log','application','thickness_log','width','country','customer','product_ref','status','item type']]
y=copper_data['selling_price_log']


# In[40]:


from sklearn.preprocessing import StandardScaler


# In[41]:


SS = StandardScaler()
SS.fit_transform(x)


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[44]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[45]:


LR = LinearRegression()
LR.fit(X_train, y_train)
print(LR.score(X_train, y_train))
print(LR.score(X_test, y_test))


# In[46]:


RF = RandomForestRegressor(n_estimators = 20, max_depth = 4, random_state = 42)
RF.fit(X_train, y_train)
print(RF.score(X_train, y_train))
print(RF.score(X_test,y_test))


# In[47]:


GBR = GradientBoostingRegressor(n_estimators = 10, learning_rate = 0.1, random_state =42)
GBR.fit(X_train,y_train)
print(GBR.score(X_train,y_train))
print(GBR.score(X_test,y_test))


# In[48]:


DTR = DecisionTreeRegressor()
# hyperparameters
param_grid = {'max_depth': [2, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['auto', 'sqrt', 'log2']}
# gridsearchcv
grid_search = GridSearchCV(estimator=DTR, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


# ###### check metrix

# In[49]:


from sklearn import metrics


# In[50]:


mse = metrics.mean_squared_error(y_test,y_pred)


# In[51]:


mse


# In[52]:


mae = metrics.mean_absolute_error(y_test,y_pred)


# In[53]:


mae


# In[54]:


r2 = metrics.r2_score(y_test,y_pred)


# In[55]:


r2


# In[56]:


print('Mean squared error:', mse)
print('R-squared:', r2)


# In[57]:


copper_industry.status.value_counts()


# In[58]:


data = copper_industry.copy()


# In[59]:


data = copper_industry[copper_industry['status'].isin(['Won', 'Lost'])]
len(data)


# In[60]:


data.status.value_counts()


# In[61]:


OE = OrdinalEncoder()
data.status = OE.fit_transform(data[['status']])
data['item type'] = OE.fit_transform(data[['item type']])


# In[62]:


X_1 = data[['quantity tons','selling_price','item type','application','thickness','width','country','customer','product_ref']]
y_1 = data['status']


# In[63]:


SS.fit_transform(X_1)          #### standard scaler


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X_1, y_1, test_size = 0.3, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[65]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# In[68]:


DTC = DecisionTreeClassifier(max_depth = 5, random_state = 1)
DTC.fit(X_train, y_train)
train_score = DTC.score(X_train, y_train)
test_score = DTC.score(X_test, y_test)
print(train_score)
print(test_score)


# In[69]:


knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))


# In[70]:


gbc = GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1,random_state = 28)
gbc.fit(X_train, y_train)
print(gbc.score(X_train, y_train))
print(gbc.score(X_test, y_test))


# In[71]:


RFC = RandomForestClassifier(n_estimators = 20, max_depth =6,random_state = 35)
RFC.fit(X_train, y_train)
print(RFC.score(X_train, y_train))
print(RFC.score(X_test, y_test))


# In[72]:


LR = LogisticRegression()
LR.fit(X_train,y_train)
print(LR.score(X_train, y_train))
print(LR.score(X_test, y_test))


# In[73]:


from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error


# In[74]:


from sklearn.ensemble import ExtraTreesClassifier
ETC = ExtraTreesClassifier(n_estimators=20, random_state=0)
ETC.fit(X_train, y_train)
print(ETC.score(X_train, y_train))
print(ETC.score(X_test, y_test))


# In[75]:


knn.predict(X_test)


# In[76]:


y_prediction= knn.predict(X_test)


# In[77]:


y_prediction


# In[78]:


from sklearn.metrics import confusion_matrix


# In[79]:


CM = confusion_matrix(y_test,y_prediction)
CM


# ###### draw heatmap

# In[80]:


plt.figure(figsize = (7,5))
sns.heatmap(CM, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[81]:


from sklearn.metrics import accuracy_score, f1_score,classification_report, roc_curve, auc


# In[82]:


accuracy_score(y_test, y_prediction)


# In[83]:


f1_score(y_test,y_prediction, average = 'macro')


# In[84]:


from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


# In[86]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_prediction))


# In[87]:


print("Classification Report:")
print(classification_report(y_test, y_prediction))


# In[88]:


fpr, tpr, thresholds = roc_curve(y_test, y_prediction)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[89]:


print(classification_report(y_test, y_prediction))


# In[ ]:





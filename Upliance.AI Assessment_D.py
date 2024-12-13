#!/usr/bin/env python
# coding: utf-8

# ## Upliance.AI Assessment

# In[4]:


# Importing the Libraries


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


user_details = pd.read_excel("user_details.xlsx")
cooking_session = pd.read_excel("cooking_session.xlsx")
order_details = pd.read_excel("order_details.xlsx")


# In[13]:


user_details


# In[14]:


cooking_session


# In[15]:


order_details


# In[19]:


# Calculate the mean of the "rating" column, ignoring NaN values
avg_rating = order_details['Rating'].mean()


# In[20]:


avg_rating


# In[21]:


# Replace NaN values with the average
order_details['Rating'] = order_details['Rating'].fillna(avg_rating)


# In[22]:


# Save the updated DataFrame back to a file
order_details.to_excel("updated_order_details.xlsx", index=False)


# In[23]:


order_details


# In[24]:


# Merge datasets
merged_data = cooking_session.merge(order_details, on="User ID").merge(user_details, on="User ID")


# In[36]:


merged_data.head(10)


# In[ ]:


# analyzing the relationship between cooking sessions and user orders


# In[28]:


# Calculate the number of cooking sessions per user
cooking_counts = merged_data.groupby('User ID')['Session ID_x'].nunique().reset_index()
cooking_counts.rename(columns={'Session ID_x': 'Cooking Session Count'}, inplace=True)


# In[29]:


cooking_counts


# In[30]:


# Calculate the number of orders per user
order_counts = merged_data.groupby('User ID')['Order ID'].nunique().reset_index()
order_counts.rename(columns={'Order ID': 'Order Count'}, inplace=True)


# In[31]:


order_counts


# In[32]:


# Merge the two metrics
user_analysis = cooking_counts.merge(order_counts, on="User ID")


# In[33]:


user_analysis


# In[34]:


# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_analysis, x='Cooking Session Count', y='Order Count')
plt.title("Relationship Between Cooking Sessions and User Orders")
plt.xlabel("Cooking Session Count")
plt.ylabel("Order Count")
plt.show()


# In[35]:


# Calculate correlation
correlation = user_analysis['Cooking Session Count'].corr(user_analysis['Order Count'])
print(f"Correlation between cooking sessions and user orders: {correlation:.2f}")


# In[37]:


# Regression Analysis


# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[39]:


# Prepare data
X = user_analysis[['Cooking Session Count']].values  # Predictor
y = user_analysis['Order Count'].values  # Target


# In[40]:


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[41]:


# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# In[42]:


# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R-squared for order count prediction: {r2:.2f}")


# In[43]:


# Display regression coefficients
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")


# In[44]:


# Plot data with regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_analysis, x='Cooking Session Count', y='Order Count')

# Regression line
x_range = np.linspace(user_analysis['Cooking Session Count'].min(), user_analysis['Cooking Session Count'].max(), 100)
y_line = model.coef_[0] * x_range + model.intercept_
plt.plot(x_range, y_line, color='red', label='Regression Line')

plt.title("Cooking Sessions vs. Order Count with Regression Line")
plt.xlabel("Cooking Session Count")
plt.ylabel("Order Count")
plt.legend()
plt.show()


# In[46]:


# Statistical Tests


# In[45]:


from scipy.stats import pearsonr

# Correlation test
corr, p_value = pearsonr(user_analysis['Cooking Session Count'], user_analysis['Order Count'])
print(f"Pearson correlation: {corr:.2f}, p-value: {p_value:.3e}")


# In[50]:


# Calculate the most popular dishes
popular_dishes = merged_data['Dish Name_x'].value_counts().head()
print("Top Popular Dishes:")
print(popular_dishes)


# In[52]:


# Analyze the influence of age groups on user ratings
# Create age groups
bins = [0, 18, 30, 45, 60, 100]
labels = ['0-18', '19-30', '31-45', '46-60', '60+']
merged_data['Age Group'] = pd.cut(merged_data['Age'], bins=bins, labels=labels)

age_group_rating = merged_data.groupby('Age Group')['Rating'].mean()
print("\nAverage Rating by Age Group:")
print(age_group_rating)


# In[60]:


# Boxplot for ratings by age group
plt.figure(figsize=(10, 6))
sns.boxplot(data=merged_data, x='Age Group', y='Rating')
plt.title("Distribution of Ratings by Age Group")
plt.show()


# In[68]:


# Additional Analysis: Regional trends in engagement
region_engagement = merged_data.groupby('Location')['Duration (mins)'].mean()
print("\nAverage Cooking Session Duration by Location:")
print(region_engagement)


# In[65]:


# Average rating by Age Group and Location
demographic_impact = merged_data.groupby(['Age Group', 'Location'])['Rating'].mean().reset_index()


# In[66]:


demographic_impact


# In[62]:


# Pivot table for better visualization
pivot_table = demographic_impact.pivot(index='Age Group', columns='Location', values='Rating')



# In[63]:


pivot_table


# In[67]:


# Visualize with heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Average Rating by Age Group and Location")
plt.xlabel("Location")
plt.ylabel("Age Group")
plt.show()


# #  visualizations to showcase key insights

# In[69]:


# 1. Relationship Between Cooking Sessions and User Orders
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_analysis, x='Cooking Session Count', y='Order Count')
plt.title("Relationship Between Cooking Sessions and User Orders")
plt.xlabel("Cooking Session Count")
plt.ylabel("Order Count")
plt.show()


# In[70]:


# 2. Correlation Between Cooking Sessions and Order Counts
correlation = user_analysis['Cooking Session Count'].corr(user_analysis['Order Count'])
print(f"Correlation between cooking sessions and user orders: {correlation:.2f}")


# In[71]:


# 3. Regression Line for Cooking Sessions vs. Order Count
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_analysis, x='Cooking Session Count', y='Order Count')

# Regression line
x_range = np.linspace(user_analysis['Cooking Session Count'].min(), user_analysis['Cooking Session Count'].max(), 100)
y_line = model.coef_[0] * x_range + model.intercept_
plt.plot(x_range, y_line, color='red', label='Regression Line')

plt.title("Cooking Sessions vs. Order Count with Regression Line")
plt.xlabel("Cooking Session Count")
plt.ylabel("Order Count")
plt.legend()
plt.show()


# In[72]:


# 4. Distribution of Ratings by Age Group
plt.figure(figsize=(10, 6))
sns.boxplot(data=merged_data, x='Age Group', y='Rating')
plt.title("Distribution of Ratings by Age Group")
plt.show()


# In[73]:


# 5. Average Cooking Session Duration by Location
plt.figure(figsize=(10, 6))
sns.barplot(x=region_engagement.index, y=region_engagement.values)
plt.title("Average Cooking Session Duration by Location")
plt.xlabel("Location")
plt.ylabel("Average Duration (mins)")
plt.show()


# In[74]:


# 6. Heatmap of Average Ratings by Age Group and Location
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Average Rating by Age Group and Location")
plt.xlabel("Location")
plt.ylabel("Age Group")
plt.show()


# In[ ]:





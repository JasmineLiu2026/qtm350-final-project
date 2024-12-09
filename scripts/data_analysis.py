# %% [markdown]
# # Importing Dataset and libraries (and some preliminary exploration to understand data)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
df_mc_gdppc = pd.read_csv("/Users/jasmineliu/Downloads/QTM350/Final project/dataset/metadata_countries.csv")
df_d_gdppc = pd.read_csv("/Users/jasmineliu/Downloads/QTM350/Final project/dataset/gdppc.csv")



# %%
df_mc_gdppc.head()

# %%
print(df_mc_gdppc.shape)

# %%
df_d_gdppc.head()

# %%
df_mc_gdppc.query('Region.isnull()') #these are the country code that are just regions instead of individual country

# %%
df_d_gdppc["Country Name"].unique()

# %%
usa_data = df_d_gdppc[df_d_gdppc['Country Name'] == 'United States']
df_d_melted = usa_data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='GDP per Capita')
df_d_melted  = df_d_melted.dropna(subset=['GDP per Capita'])
df_d_melted['Year'] = df_d_melted['Year'].astype(int)
df_d_melted = df_d_melted.sort_values('Year')
df_d_melted 


# %%
plt.figure(figsize=(14, 7))
plt.plot(df_d_melted['Year'], df_d_melted['GDP per Capita'], marker='o', linestyle='-', markersize=4)
plt.title("United States GDP per Capita (Constant 2015 US$)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("GDP per Capita (Constant 2015 US$)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# %%
china_data = df_d_gdppc[df_d_gdppc['Country Name'] == 'China']
df_d_melted = china_data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='GDP per Capita')
df_d_melted  = df_d_melted.dropna(subset=['GDP per Capita'])
df_d_melted['Year'] = df_d_melted['Year'].astype(int)
df_d_melted = df_d_melted.sort_values('Year')
df_d_melted 

# %%
plt.figure(figsize=(14, 7))
plt.plot(df_d_melted['Year'], df_d_melted['GDP per Capita'], marker='o', linestyle='-', markersize=4)
plt.title("China GDP per Capita (Constant 2015 US$)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("GDP per Capita (Constant 2015 US$)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# %%
india_data = df_d_gdppc[df_d_gdppc['Country Name'] == 'India']
df_d_melted = india_data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='GDP per Capita')
df_d_melted  = df_d_melted.dropna(subset=['GDP per Capita'])
df_d_melted['Year'] = df_d_melted['Year'].astype(int)
df_d_melted = df_d_melted.sort_values('Year')
df_d_melted 

# %%
plt.figure(figsize=(14, 7))
plt.plot(df_d_melted['Year'], df_d_melted['GDP per Capita'], marker='o', linestyle='-', markersize=4)
plt.title("India GDP per Capita (Constant 2015 US$)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("GDP per Capita (Constant 2015 US$)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# %%
uk_data = df_d_gdppc[df_d_gdppc['Country Name'] == 'United Kingdom']
df_d_melted = uk_data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='GDP per Capita')
df_d_melted  = df_d_melted.dropna(subset=['GDP per Capita'])
df_d_melted['Year'] = df_d_melted['Year'].astype(int)
df_d_melted = df_d_melted.sort_values('Year')
df_d_melted 

# %%
plt.figure(figsize=(14, 7))
plt.plot(df_d_melted['Year'], df_d_melted['GDP per Capita'], marker='o', linestyle='-', markersize=4)
plt.title("UK GDP per Capita (Constant 2015 US$)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("GDP per Capita (Constant 2015 US$)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# %%
df_mc_gdpg = pd.read_csv("/Users/jasmineliu/Downloads/QTM350/Final project/dataset/metadata_countries.csv")
df_d_gdpg = pd.read_csv("/Users/jasmineliu/Downloads/QTM350/Final project/dataset/gdpg.csv")


# %%
usa_data_2 = df_d_gdpg[df_d_gdpg['Country Name'] == 'United States']
df_d_melted_2 = usa_data_2.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='GDP Growth')
df_d_melted_2  = df_d_melted_2.dropna(subset=['GDP Growth'])
df_d_melted_2['Year'] = df_d_melted_2['Year'].astype(int)
df_d_melted_2 = df_d_melted_2.sort_values('Year')
df_d_melted_2

# %%
plt.figure(figsize=(14, 7))
plt.bar(df_d_melted_2['Year'], df_d_melted_2['GDP Growth'])
plt.title("USA GDP Growth", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("GDP Growth)", fontsize=12)

# %%
df_mc_e = pd.read_csv("/Users/jasmineliu/Downloads/QTM350/Final project/dataset/metadata_countries.csv")
df_d_e = pd.read_csv("/Users/jasmineliu/Downloads/QTM350/Final project/dataset/employment.csv")

# %%
df_mc_e.head()

# %%
df_d_e.head()

# %%
usa_data_3 = df_d_e[df_d_e['Country Name'] == 'United States']
df_d_melted_3 = usa_data_3.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='Employment to Population Ratio')
df_d_melted_3  = df_d_melted_3.dropna(subset=['Employment to Population Ratio'])
df_d_melted_3['Year'] = df_d_melted_3['Year'].astype(int)
df_d_melted_3 = df_d_melted_3.sort_values('Year')
df_d_melted_3

# %%
plt.figure(figsize=(14, 7))
plt.plot(df_d_melted_3['Year'], df_d_melted_3['Employment to Population Ratio'])
plt.title("US Employment to Population Ratio", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Employment to Population Ratio)", fontsize=12)

# %% [markdown]
# ## Merging income group label with all three metadata based on country code.
# This produces 3 data consiting of country name, country code, income group, and their original features.

# %%
merged_data_gdppc = pd.merge(df_d_gdppc, df_mc_gdppc, on="Country Code", how="left")
merged_data_gdppc.head()

# %%
merged_data_e = pd.merge(df_d_e, df_mc_e, on="Country Code", how="left")
merged_data_e.head()

# %%
merged_data_gdpg = pd.merge(df_d_gdpg, df_mc_gdpg, on="Country Code", how="left")
merged_data_gdpg.head()

# %% [markdown]
# ### Some exploratory analysis:

# %%
grouped_count = merged_data_gdppc.groupby("IncomeGroup").size()
print(grouped_count)

# %%
# Filter data for only 'High income' group
high_income_data = merged_data_gdppc[merged_data_gdppc["IncomeGroup"] == "High income"]
high_income_data["Country Name"].unique()

# %%
# Filter data for only 'low income' group
low_income_data = merged_data_gdppc[merged_data_gdppc["IncomeGroup"] == "Low income"]
low_income_data["Country Name"].unique()

# %%
afghan_data= df_d_e[df_d_e['Country Name'] == 'Afghanistan']
afghan_d_melted = afghan_data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='Employment to Population Ratio')
afghan_d_melted  = afghan_d_melted.dropna(subset=['Employment to Population Ratio'])
afghan_d_melted['Year'] = afghan_d_melted['Year'].astype(int)
afghan_d_melted = afghan_d_melted.sort_values('Year')

# %%
plt.figure(figsize=(14, 7))
plt.plot(afghan_d_melted['Year'], afghan_d_melted['Employment to Population Ratio'])
plt.title("Afghanistan Employment to Population Ratio", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Employment to Population Ratio)", fontsize=12)

# %%
afghan_d_melted.sort_values(by="Year", ascending=False)

# %%
afghan_data= df_d_gdppc[df_d_e['Country Name'] == 'Afghanistan']
afghan_d_melted = afghan_data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='GDP per Capita')
afghan_d_melted  = afghan_d_melted.dropna(subset=['GDP per Capita'])
afghan_d_melted['Year'] = afghan_d_melted['Year'].astype(int)
afghan_d_melted = afghan_d_melted.sort_values('Year')

# %%
plt.figure(figsize=(14, 7))
plt.plot(afghan_d_melted['Year'], afghan_d_melted['GDP per Capita'])
plt.title("Afghanistan GDP per Capita", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("GDP per Capita", fontsize=12)

# %%
USA_data= df_d_e[df_d_e['Country Name'] == 'United States']
USA_d_melted = USA_data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='Employment to Population Ratio')
USA_d_melted  = USA_d_melted.dropna(subset=['Employment to Population Ratio'])
USA_d_melted['Year'] = USA_d_melted['Year'].astype(int)
USA_d_melted = USA_d_melted.sort_values('Year')

# %%
plt.figure(figsize=(14, 7))
plt.plot(USA_d_melted['Year'], USA_d_melted['Employment to Population Ratio'])
plt.title("USA Employment to Population Ratio", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Employment to Population Ratio)", fontsize=12)

# %%
USA_data= df_d_gdpg[df_d_gdpg['Country Name'] == 'United States']
USA_d_melted = USA_data.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], 
                    var_name='Year', 
                    value_name='GDP Growth')
USA_d_melted  = USA_d_melted.dropna(subset=['GDP Growth'])
USA_d_melted['Year'] = USA_d_melted['Year'].astype(int)
USA_d_melted = USA_d_melted.sort_values('Year')

# %%
plt.figure(figsize=(14, 7))
plt.plot(USA_d_melted['Year'], USA_d_melted['GDP Growth'])
plt.title("USA GDP Growth", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("GDP Growth", fontsize=12)

# %% [markdown]
# ## Preprocessing:

# %% [markdown]
# ### Preprocess the dataset for Employment to Population Ratio Values

# %%
merged_data_e.head()

# %%
merged_data_e = merged_data_e.drop(columns=['Indicator Name', 'Indicator Code', 'SpecialNotes', 'TableName'])
melted_data_e = merged_data_e.melt(
    id_vars=['Country Name', 'Country Code', 'Region', 'IncomeGroup'], 
    var_name='Year', 
    value_name='Employment to Population Ratio'
)

#drop missing values
melted_data_e = melted_data_e.dropna(subset=['Employment to Population Ratio'])

melted_data_e['Year'] = melted_data_e['Year'].astype(int)

melted_data_e = melted_data_e.sort_values(['Country Name', 'Year'])

#check if theres any remaining missing values in other columns
print(melted_data_e.isnull().sum())
melted_data_e.head()

# %% [markdown]
# ### Preprocess the data for GDP Growth

# %%
merged_data_gdpg = merged_data_gdpg.drop(columns=['Indicator Name', 'Indicator Code', 'SpecialNotes', 'TableName'])
melted_data_growth = merged_data_gdpg.melt(
    id_vars=['Country Name', 'Country Code', 'Region', 'IncomeGroup'], 
    var_name='Year', 
    value_name='GDP Growth'
)

melted_data_growth = melted_data_growth.dropna(subset=['GDP Growth'])

melted_data_growth['Year'] = melted_data_growth['Year'].astype(int)

melted_data_growth = melted_data_growth.sort_values(['Country Name', 'Year'])

print(melted_data_growth.isnull().sum())
melted_data_growth.head()


# %% [markdown]
# ### Preprocess the data for GDP per Capita

# %%
merged_data_gdppc = merged_data_gdppc.drop(columns=['Indicator Name', 'Indicator Code', 'SpecialNotes', 'TableName'])
melted_data_gdppc = merged_data_gdppc.melt(
    id_vars=['Country Name', 'Country Code', 'Region', 'IncomeGroup'], 
    var_name='Year', 
    value_name='GDP per Capita'
)

melted_data_gdppc = melted_data_gdppc.dropna(subset=['GDP per Capita'])

melted_data_gdppc['Year'] = melted_data_gdppc['Year'].astype(int)

melted_data_gdppc = melted_data_gdppc.sort_values(['Country Name', 'Year'])

print(melted_data_gdppc.isnull().sum())
print(melted_data_gdppc.shape)
melted_data_gdppc.head()

# %% [markdown]
# ## Done preprocessing! Now we want to generate some summary of key variables

# %% [markdown]
# Employment to population ratio data:

# %%
print(melted_data_e.info())

print(melted_data_e.describe())

# %%
print(melted_data_gdppc.info())

print(melted_data_gdppc.describe())


# %%
print(melted_data_growth.info())

print(melted_data_growth.describe())

# %% [markdown]
# ## Analysis:

# %% [markdown]
# Research questions: How do GDP growth rates differ between high-income and low-income countries?
# How does a employment-to-population ratio correlate with GDP per capita in different income groups?
# 

# %% [markdown]
# #### Comparing GDP Growth across income groups:

# %%
gdp_growth_by_income_group = melted_data_growth.groupby(["IncomeGroup", 'Year'])['GDP Growth'].mean().reset_index()

# %%
gdp_growth_by_income_group

# %%
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.lineplot(data=gdp_growth_by_income_group, x='Year', y='GDP Growth', hue='IncomeGroup', marker='o')
plt.title('Average GDP Growth by Income Group (1960-2023)')
plt.xlabel('Year')
plt.ylabel('Average GDP Growth (%)')
plt.legend(title='Income Group')
plt.savefig('Average GDP Growth by Income Group.png') 
plt.show()

# %% [markdown]
# Thee was a 2008 global financial crisis and we see that across all groups expereinced a dip in GDP growth around 2008-2010. Worth noticing is that low and lower middle income groups did not experience a drastic decrease in GDP growth like the high and upper middle income groups. This is likely because many low-income countries have economies that are less reliant on global financial markets, making them less vulnerable to the direct effects of financial crises originating in high-income regions. 
# 
# In 2020, the COVID-19 pandemic caused significant economic disruption across all income groups. The upper-middle-income group experienced the most drastic decrease in average GDP growth within this period, likely due to their higher exposure to global supply chain disruptions, service sectors (such as tourism, aviation, and hospitality), and the lockdown measures in many developed economies. Overall, despite the varying degrees of impact, all income groups displayed similar trends, with low income groups exhibiting the most fluctuations.

# %%
merged_gdppc_e = pd.merge(melted_data_gdppc, melted_data_e, 
                       on=['Country Name', 'Year'], 
                       how='inner')  # 'inner' join to keep only matching rows

merged_gdppc_e = merged_gdppc_e.rename(columns={
    'Country Code_x' : 'Country Code',
    'IncomeGroup_x' : 'IncomeGroup',
    'Region_x': 'Region'
})

# Drop the unnecessary columns
merged_gdppc_e = merged_gdppc_e.drop(columns=['Country Code_y', 'IncomeGroup_y', 'Region_y'])

merged_gdppc_e  = merged_gdppc_e.dropna()

# %%
merged_gdppc_e.head()

# %%
# Group by Income Group and we want to calculate correlation between employment ratio and gdp per capita for each group
correlation_results = []
for income_group in  merged_gdppc_e['IncomeGroup'].unique():
    group_data =  merged_gdppc_e[ merged_gdppc_e['IncomeGroup'] == income_group]
    correlation = group_data['GDP per Capita'].corr(group_data['Employment to Population Ratio'])
    correlation_results.append([income_group, correlation])
    print(f'Correlation for {income_group} group: {correlation:.3f}')


# %%
correlation_df = pd.DataFrame(correlation_results, columns=['IncomeGroup', 'Correlation'])
correlation_df.to_csv('income_group_correlation.csv', index=False)

# %% [markdown]
# Our correlation result imply that, in low-income countries, a higher employment rate does not necessarily correlate with higher economic output. Low-income countries often rely on labor-intensive industries and agriculture, where high employment doesn’t always lead to a proportional increase in GDP. This could be due to the nature of the informal labor market, which doesn't contribute significantly to GDP in the same way formal sector jobs do like in other higher income countries. 
# 
# The very low negative correlation of -0.042 indicates almost no relationship between GDP per capita and Employment-to-Population Ratio in upper-middle-income countries. This suggests that fluctuations in employment levels do not strongly influence economic output in this group.
# 
# Lower middle income group resemble similar correlation like the low income group, possible due to similar reasons.
# 
# The positive correlation of 0.444 in high-income countries suggests a moderate positive relationship between GDP per capita and Employment-to-Population Ratio. In other words, as the employment ratio increases, GDP per capita tends to increase as well.
# 
# High-income countries often have more developed, diversified, and efficient economies where higher employment levels (especially in the formal sector) directly contribute to greater productivity and economic output. For these countries, a higher employment-to-population ratio likely signals an efficient utilization of human resources, leading to higher GDP per capita.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Group by Year and IncomeGroup, calculate the mean GDP per capita
gdp_trends = merged_gdppc_e.groupby(['Year', 'IncomeGroup'])['GDP per Capita'].mean().reset_index()

# Plot the trends
plt.figure(figsize=(10, 6))
sns.lineplot(data=gdp_trends, x='Year', y='GDP per Capita', hue='IncomeGroup', marker='o')
plt.title('Trend of GDP per Capita by Income Group (1990-2023)')
plt.xlabel('Year')
plt.ylabel('GDP per Capita (USD)')
plt.legend(title='Income Group')
plt.grid(True)
plt.savefig('Trend of GDP per Capita by Income Group.png')  
plt.show()


# %%
# Group by Year and IncomeGroup, calculate the mean Employment to Population Ratio
employment_trends = merged_gdppc_e.groupby(['Year', 'IncomeGroup'])['Employment to Population Ratio'].mean().reset_index()

# Plot the trends
plt.figure(figsize=(10, 6))
sns.lineplot(data=employment_trends, x='Year', y='Employment to Population Ratio', hue='IncomeGroup', marker='o')
plt.title('Trend of Employment to Population Ratio by Income Group (1990-2023)')
plt.xlabel('Year')
plt.ylabel('Employment to Population Ratio')
plt.legend(title='Income Group')
plt.grid(True)
plt.savefig('Trend of Employment to Population Ratio by Income Group.png')  
plt.show()
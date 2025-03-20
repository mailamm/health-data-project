```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
```

## 1. Data Source and Preparation

#### Loading the Dataset
- The dataset is loaded from a CSV file for analysis.
- **Metadata columns** (used for row identification) are retained.

#### Selecting Relevant Features
- **Health outcome columns** are dropped, except for **Premature Death**, which is used as the target variable.
- **Additional measures** are removed as per the assignment requirements.
- Columns that do not contribute meaningful information to the analysis are also dropped.

#### Data Type Conversion
- **Raw_value columns** are converted to **numerical format** to facilitate calculations in **Exploratory Data Analysis (EDA)** and further modeling.



```python
# Load the dataset while skipping the first descriptive row
file_path = 'analytic_data2024.csv'
df = pd.read_csv(file_path,dtype=str, header=[1])
print("Data shape: ", df.shape)
```

    Data shape:  (3195, 770)



```python
# List of metadata columns to retain
metadata_columns = ["statecode", "countycode", "fipscode", "state", "county", "year", "county_clustered"]

# List of Health Outcomes to exclude except for Premature Death
health_outcomes = [
    "v002_rawvalue", # Poor or Fair Health
    "v036_rawvalue",  # Poor Physical Health Days
    "v042_rawvalue",  # Poor Mental Health Days
    "v037_rawvalue",  # Low Birthweight
]

# List of Additional Measures to exclude
additional_measures = [
    "v147_rawvalue",  # Life Expectancy
    "v127_rawvalue",  # Premature Age-Adjusted Mortality
    "v128_rawvalue",  # Child Mortality
    "v129_rawvalue",  # Infant Mortality
    "v144_rawvalue",  # Frequent Physical Distress
    "v145_rawvalue",  # Frequent Mental Distress
    "v060_rawvalue",  # Diabetes Prevalence
    "v061_rawvalue",  # HIV Prevalence
    "v139_rawvalue",  # Food Insecurity
    "v083_rawvalue",  # Limited Access to Healthy Foods
    "v138_rawvalue",  # Drug Overdose Deaths
    "v143_rawvalue",  # Insufficient Sleep
    "v003_rawvalue",  # Uninsured Adults
    "v122_rawvalue",  # Uninsured Children
    "v131_rawvalue",  # Other Primary Care Providers
    "v021_rawvalue",  # High School Graduation
    "v149_rawvalue",  # Disconnected Youth
    "v159_rawvalue",  # Reading Scores
    "v160_rawvalue",  # Math Scores
    "v167_rawvalue",  # School Segregation
    "v169_rawvalue",  # School Funding Adequacy
    "v151_rawvalue",  # Gender Pay Gap
    "v063_rawvalue",  # Median Household Income
    "v170_rawvalue",  # Living Wage
    "v065_rawvalue",  # Children Eligible for Free or Reduced Price Lunch
    "v141_rawvalue",  # Residential Segregation - Black/White
    "v171_rawvalue",  # Child Care Cost Burden
    "v172_rawvalue",  # Child Care Centers
    "v015_rawvalue",  # Homicides
    "v161_rawvalue",  # Suicides
    "v148_rawvalue",  # Firearm Fatalities
    "v039_rawvalue",  # Motor Vehicle Crash Deaths
    "v158_rawvalue",  # Juvenile Arrests
    "v177_rawvalue",  # Voter Turnout
    "v178_rawvalue",  # Census Participation
    "v156_rawvalue",  # Traffic Volume
    "v153_rawvalue",  # Homeownership
    "v154_rawvalue",  # Severe Housing Cost Burden
    "v166_rawvalue",  # Broadband Access
    "v051_rawvalue",  # Population
    "v052_rawvalue",  # % Below 18 Years of Age
    "v053_rawvalue",  # % 65 and Older
    "v054_rawvalue",  # % Non-Hispanic Black
    "v055_rawvalue",  # % American Indian or Alaska Native
    "v081_rawvalue",  # % Asian
    "v080_rawvalue",  # % Native Hawaiian or Other Pacific Islander
    "v056_rawvalue",  # % Hispanic
    "v126_rawvalue",  # % Non-Hispanic White
    "v059_rawvalue",  # % Not Proficient in English
    "v057_rawvalue",  # % Female
    "v058_rawvalue",  # % Rural
]

#Select columns that contain "rawvalue" but exclude health outcomes and additional measures
rawvalue_columns = [col for col in df.columns if "rawvalue" in col]
selected_features = [col for col in rawvalue_columns if col not in health_outcomes and col not in additional_measures]

# New DataFrame with only selected features and metadata columns
df_filtered = df[metadata_columns + selected_features]

df_filtered.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statecode</th>
      <th>countycode</th>
      <th>fipscode</th>
      <th>state</th>
      <th>county</th>
      <th>year</th>
      <th>county_clustered</th>
      <th>v001_rawvalue</th>
      <th>v009_rawvalue</th>
      <th>v011_rawvalue</th>
      <th>...</th>
      <th>v024_rawvalue</th>
      <th>v044_rawvalue</th>
      <th>v082_rawvalue</th>
      <th>v140_rawvalue</th>
      <th>v135_rawvalue</th>
      <th>v125_rawvalue</th>
      <th>v124_rawvalue</th>
      <th>v136_rawvalue</th>
      <th>v067_rawvalue</th>
      <th>v137_rawvalue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00</td>
      <td>000</td>
      <td>00000</td>
      <td>US</td>
      <td>United States</td>
      <td>2024</td>
      <td>NaN</td>
      <td>7971.5097891</td>
      <td>0.15</td>
      <td>0.34</td>
      <td>...</td>
      <td>0.163</td>
      <td>4.9028422933</td>
      <td>0.2493158226</td>
      <td>9.0856186518</td>
      <td>80.005527999</td>
      <td>7.35</td>
      <td>NaN</td>
      <td>0.1673889139</td>
      <td>0.7167332819</td>
      <td>0.364</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01</td>
      <td>000</td>
      <td>01000</td>
      <td>AL</td>
      <td>Alabama</td>
      <td>2024</td>
      <td>NaN</td>
      <td>11415.734833</td>
      <td>0.179</td>
      <td>0.406</td>
      <td>...</td>
      <td>0.218</td>
      <td>5.212687998</td>
      <td>0.3058752993</td>
      <td>11.694729852</td>
      <td>90.379698685</td>
      <td>9.3</td>
      <td>0.223880597</td>
      <td>0.1308124681</td>
      <td>0.8282502403</td>
      <td>0.352</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01</td>
      <td>001</td>
      <td>01001</td>
      <td>AL</td>
      <td>Autauga County</td>
      <td>2024</td>
      <td>1</td>
      <td>9407.9484384</td>
      <td>0.169</td>
      <td>0.389</td>
      <td>...</td>
      <td>0.157</td>
      <td>4.6376753942</td>
      <td>0.2270250636</td>
      <td>12.691429055</td>
      <td>68.033478141</td>
      <td>10</td>
      <td>0</td>
      <td>0.1537569573</td>
      <td>0.8542784211</td>
      <td>0.429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01</td>
      <td>003</td>
      <td>01003</td>
      <td>AL</td>
      <td>Baldwin County</td>
      <td>2024</td>
      <td>1</td>
      <td>8981.5753533</td>
      <td>0.15</td>
      <td>0.372</td>
      <td>...</td>
      <td>0.161</td>
      <td>4.4824492469</td>
      <td>0.1905128725</td>
      <td>9.6533970764</td>
      <td>77.507984659</td>
      <td>7.6</td>
      <td>1</td>
      <td>0.1242786602</td>
      <td>0.8134699854</td>
      <td>0.379</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01</td>
      <td>005</td>
      <td>01005</td>
      <td>AL</td>
      <td>Barbour County</td>
      <td>2024</td>
      <td>1</td>
      <td>13138.848362</td>
      <td>0.25</td>
      <td>0.434</td>
      <td>...</td>
      <td>0.377</td>
      <td>5.5767967895</td>
      <td>0.5066045066</td>
      <td>8.4121134434</td>
      <td>85.215853364</td>
      <td>9.4</td>
      <td>0</td>
      <td>0.150751073</td>
      <td>0.8205251169</td>
      <td>0.367</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
# Unique numbers of value in each column
unique_counts = df_filtered.nunique()
print(unique_counts)
```

    statecode             52
    countycode           326
    fipscode            3195
    state                 52
    county              1929
    year                   1
    county_clustered       2
    v001_rawvalue       3139
    v009_rawvalue        236
    v011_rawvalue        271
    v133_rawvalue         82
    v070_rawvalue        280
    v132_rawvalue       3111
    v049_rawvalue       3194
    v134_rawvalue        912
    v045_rawvalue       2419
    v014_rawvalue       2953
    v085_rawvalue       3190
    v004_rawvalue       2942
    v088_rawvalue       2934
    v062_rawvalue       2971
    v005_rawvalue       2105
    v050_rawvalue         54
    v155_rawvalue         64
    v168_rawvalue       3189
    v069_rawvalue       3185
    v023_rawvalue       3181
    v024_rawvalue        403
    v044_rawvalue       3179
    v082_rawvalue       3179
    v140_rawvalue       3013
    v135_rawvalue       3086
    v125_rawvalue        113
    v124_rawvalue         50
    v136_rawvalue       3088
    v067_rawvalue       3188
    v137_rawvalue        573
    dtype: int64



```python
# Check data type of each column
print(df_filtered.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3195 entries, 0 to 3194
    Data columns (total 37 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   statecode         3195 non-null   object
     1   countycode        3195 non-null   object
     2   fipscode          3195 non-null   object
     3   state             3195 non-null   object
     4   county            3195 non-null   object
     5   year              3195 non-null   object
     6   county_clustered  3143 non-null   object
     7   v001_rawvalue     3140 non-null   object
     8   v009_rawvalue     3195 non-null   object
     9   v011_rawvalue     3195 non-null   object
     10  v133_rawvalue     3160 non-null   object
     11  v070_rawvalue     3195 non-null   object
     12  v132_rawvalue     3150 non-null   object
     13  v049_rawvalue     3195 non-null   object
     14  v134_rawvalue     3168 non-null   object
     15  v045_rawvalue     3049 non-null   object
     16  v014_rawvalue     2979 non-null   object
     17  v085_rawvalue     3194 non-null   object
     18  v004_rawvalue     3038 non-null   object
     19  v088_rawvalue     3108 non-null   object
     20  v062_rawvalue     3010 non-null   object
     21  v005_rawvalue     3125 non-null   object
     22  v050_rawvalue     3173 non-null   object
     23  v155_rawvalue     3175 non-null   object
     24  v168_rawvalue     3195 non-null   object
     25  v069_rawvalue     3195 non-null   object
     26  v023_rawvalue     3194 non-null   object
     27  v024_rawvalue     3194 non-null   object
     28  v044_rawvalue     3180 non-null   object
     29  v082_rawvalue     3194 non-null   object
     30  v140_rawvalue     3195 non-null   object
     31  v135_rawvalue     3089 non-null   object
     32  v125_rawvalue     3167 non-null   object
     33  v124_rawvalue     3141 non-null   object
     34  v136_rawvalue     3195 non-null   object
     35  v067_rawvalue     3195 non-null   object
     36  v137_rawvalue     3195 non-null   object
    dtypes: object(37)
    memory usage: 923.7+ KB
    None



```python
# Drop 'year' column because it contains only 2024
df_filtered = df_filtered.drop(columns=['year'])
```


```python
# Drop rows where 'county_clustered' is missing (only keeping county-level data)
data = df_filtered.dropna(subset=['county_clustered'])
```


```python
# Confirm that only county-level rows are kept
print(data['county_clustered'].isnull().sum())
```

    0



```python
# Change the data type of "rawvalue" column to numerical
data = data.copy()
raw_value_columns = [col for col in data.columns if '_rawvalue' in col]
data[raw_value_columns] = data[raw_value_columns].apply(pd.to_numeric, errors='coerce')
print(data.dtypes)
```

    statecode            object
    countycode           object
    fipscode             object
    state                object
    county               object
    county_clustered     object
    v001_rawvalue       float64
    v009_rawvalue       float64
    v011_rawvalue       float64
    v133_rawvalue       float64
    v070_rawvalue       float64
    v132_rawvalue       float64
    v049_rawvalue       float64
    v134_rawvalue       float64
    v045_rawvalue       float64
    v014_rawvalue       float64
    v085_rawvalue       float64
    v004_rawvalue       float64
    v088_rawvalue       float64
    v062_rawvalue       float64
    v005_rawvalue       float64
    v050_rawvalue       float64
    v155_rawvalue       float64
    v168_rawvalue       float64
    v069_rawvalue       float64
    v023_rawvalue       float64
    v024_rawvalue       float64
    v044_rawvalue       float64
    v082_rawvalue       float64
    v140_rawvalue       float64
    v135_rawvalue       float64
    v125_rawvalue       float64
    v124_rawvalue       float64
    v136_rawvalue       float64
    v067_rawvalue       float64
    v137_rawvalue       float64
    dtype: object



```python
# Mapping Raw Value Columns to Feature Names
feature_mapping = {
    "v001_rawvalue": "Premature Death",
    "v009_rawvalue": "Adult Smoking",
    "v011_rawvalue": "Adult Obesity",
    "v133_rawvalue": "Food Environment Index",
    "v070_rawvalue": "Physical Inactivity",
    "v132_rawvalue": "Access to Exercise Opportunities",
    "v049_rawvalue": "Excessive Drinking",
    "v134_rawvalue": "Alcohol-Impaired Driving Deaths",
    "v045_rawvalue": "Sexually Transmitted Infections",
    "v014_rawvalue": "Teen Births",
    "v085_rawvalue": "Uninsured",
    "v004_rawvalue": "Primary Care Physicians",
    "v088_rawvalue": "Dentists",
    "v062_rawvalue": "Mental Health Providers",
    "v005_rawvalue": "Preventable Hospital Stays",
    "v050_rawvalue": "Mammography Screening",
    "v155_rawvalue": "Flu Vaccinations",
    "v168_rawvalue": "High School Completion",
    "v069_rawvalue": "Some College",
    "v023_rawvalue": "Unemployment",
    "v024_rawvalue": "Children in Poverty",
    "v044_rawvalue": "Income Inequality",
    "v082_rawvalue": "Children in Single-Parent Households",
    "v140_rawvalue": "Social Associations",
    "v135_rawvalue": "Injury Deaths",
    "v125_rawvalue": "Air Pollution - Particulate Matter",
    "v124_rawvalue": "Drinking Water Violations",
    "v136_rawvalue": "Severe Housing Problems",
    "v067_rawvalue": "Driving Alone to Work",
    "v137_rawvalue": "Long Commute - Driving Alone"
}

# Rename Columns
data = data.rename(columns=feature_mapping)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>statecode</th>
      <th>countycode</th>
      <th>fipscode</th>
      <th>state</th>
      <th>county</th>
      <th>county_clustered</th>
      <th>Premature Death</th>
      <th>Adult Smoking</th>
      <th>Adult Obesity</th>
      <th>Food Environment Index</th>
      <th>...</th>
      <th>Children in Poverty</th>
      <th>Income Inequality</th>
      <th>Children in Single-Parent Households</th>
      <th>Social Associations</th>
      <th>Injury Deaths</th>
      <th>Air Pollution - Particulate Matter</th>
      <th>Drinking Water Violations</th>
      <th>Severe Housing Problems</th>
      <th>Driving Alone to Work</th>
      <th>Long Commute - Driving Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>01</td>
      <td>001</td>
      <td>01001</td>
      <td>AL</td>
      <td>Autauga County</td>
      <td>1</td>
      <td>9407.948438</td>
      <td>0.169</td>
      <td>0.389</td>
      <td>6.7</td>
      <td>...</td>
      <td>0.157</td>
      <td>4.637675</td>
      <td>0.227025</td>
      <td>12.691429</td>
      <td>68.033478</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.153757</td>
      <td>0.854278</td>
      <td>0.429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01</td>
      <td>003</td>
      <td>01003</td>
      <td>AL</td>
      <td>Baldwin County</td>
      <td>1</td>
      <td>8981.575353</td>
      <td>0.150</td>
      <td>0.372</td>
      <td>7.5</td>
      <td>...</td>
      <td>0.161</td>
      <td>4.482449</td>
      <td>0.190513</td>
      <td>9.653397</td>
      <td>77.507985</td>
      <td>7.6</td>
      <td>1.0</td>
      <td>0.124279</td>
      <td>0.813470</td>
      <td>0.379</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01</td>
      <td>005</td>
      <td>01005</td>
      <td>AL</td>
      <td>Barbour County</td>
      <td>1</td>
      <td>13138.848362</td>
      <td>0.250</td>
      <td>0.434</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.377</td>
      <td>5.576797</td>
      <td>0.506605</td>
      <td>8.412113</td>
      <td>85.215853</td>
      <td>9.4</td>
      <td>0.0</td>
      <td>0.150751</td>
      <td>0.820525</td>
      <td>0.367</td>
    </tr>
    <tr>
      <th>5</th>
      <td>01</td>
      <td>007</td>
      <td>01007</td>
      <td>AL</td>
      <td>Bibb County</td>
      <td>1</td>
      <td>12675.434581</td>
      <td>0.220</td>
      <td>0.396</td>
      <td>7.6</td>
      <td>...</td>
      <td>0.255</td>
      <td>5.669237</td>
      <td>0.307134</td>
      <td>8.897985</td>
      <td>99.933081</td>
      <td>9.8</td>
      <td>0.0</td>
      <td>0.122590</td>
      <td>0.879836</td>
      <td>0.538</td>
    </tr>
    <tr>
      <th>6</th>
      <td>01</td>
      <td>009</td>
      <td>01009</td>
      <td>AL</td>
      <td>Blount County</td>
      <td>1</td>
      <td>11541.495069</td>
      <td>0.196</td>
      <td>0.377</td>
      <td>7.7</td>
      <td>...</td>
      <td>0.158</td>
      <td>4.611946</td>
      <td>0.229695</td>
      <td>7.621822</td>
      <td>97.729173</td>
      <td>9.6</td>
      <td>0.0</td>
      <td>0.106579</td>
      <td>0.848947</td>
      <td>0.606</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




## 2. Exploratory Data Analysis (EDA)

##### Handling Missing Data
- Columns with more than **10% missing values** are dropped from the dataset.
- For columns with **less than 10% missing values**, missing values are imputed using the **median** to preserve data distribution.

##### Identifying Outliers
- **Boxplots** are used to visualize potential outliers in each feature.
- **Interquartile Range (IQR) method** is applied to detect and count the number of outliers.
- Extreme values are handled using **Winsorization**, which scales back extreme data points while maintaining overall distribution.

##### Computing Summary Statistics
- The `data.describe()` function is used to compute key statistics:
  - **Numerical features**: Includes **mean, median, standard deviation, min/max, and quartiles**.
  - **Categorical features**: Includes **counts, unique values, and most frequent category** using `data.describe(include='object')`.

### Handle missing data


```python
# Check for missing values
missing_values = data.isnull().sum()
missing_columns = missing_values[missing_values > 0].index
missing_columns
```




    Index(['Premature Death', 'Food Environment Index',
           'Access to Exercise Opportunities', 'Alcohol-Impaired Driving Deaths',
           'Sexually Transmitted Infections', 'Teen Births', 'Uninsured',
           'Primary Care Physicians', 'Dentists', 'Mental Health Providers',
           'Preventable Hospital Stays', 'Mammography Screening',
           'Flu Vaccinations', 'Unemployment', 'Children in Poverty',
           'Income Inequality', 'Children in Single-Parent Households',
           'Injury Deaths', 'Air Pollution - Particulate Matter',
           'Drinking Water Violations'],
          dtype='object')



Since no columns exceed the 10% missing threshold, we will impute all missing values using the median to avoid skewing the data.


```python
# Impute missing values using median
num_vars = data.select_dtypes(include=[np.number]).columns.tolist()
num_imputer = SimpleImputer(strategy='median')
data[num_vars] = num_imputer.fit_transform(data[num_vars])
print("Remaining missing values after imputation:\n", data.isnull().sum())
```

    Remaining missing values after imputation:
     statecode                               0
    countycode                              0
    fipscode                                0
    state                                   0
    county                                  0
    county_clustered                        0
    Premature Death                         0
    Adult Smoking                           0
    Adult Obesity                           0
    Food Environment Index                  0
    Physical Inactivity                     0
    Access to Exercise Opportunities        0
    Excessive Drinking                      0
    Alcohol-Impaired Driving Deaths         0
    Sexually Transmitted Infections         0
    Teen Births                             0
    Uninsured                               0
    Primary Care Physicians                 0
    Dentists                                0
    Mental Health Providers                 0
    Preventable Hospital Stays              0
    Mammography Screening                   0
    Flu Vaccinations                        0
    High School Completion                  0
    Some College                            0
    Unemployment                            0
    Children in Poverty                     0
    Income Inequality                       0
    Children in Single-Parent Households    0
    Social Associations                     0
    Injury Deaths                           0
    Air Pollution - Particulate Matter      0
    Drinking Water Violations               0
    Severe Housing Problems                 0
    Driving Alone to Work                   0
    Long Commute - Driving Alone            0
    dtype: int64


### Identify outliers


```python
# Visualizing outliers using boxplots
plt.figure(figsize=(15, 8))
data[num_vars].boxplot(rot=90)
plt.title("Boxplot of Features to Identify Outliers")
plt.show()
```


    
![png](IAI_HW1_mqlam_files/IAI_HW1_mqlam_19_0.png)
    



```python
# Detect outliers using IQR
Q1 = data[num_vars].quantile(0.25)
Q3 = data[num_vars].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (data[num_vars] < (Q1 - 1.5 * IQR)) | (data[num_vars] > (Q3 + 1.5 * IQR))
outliers = outlier_mask.sum()
print("Number of outliers per feature:\n", outliers)
```

    Number of outliers per feature:
     Premature Death                          71
    Adult Smoking                            52
    Adult Obesity                           114
    Food Environment Index                   85
    Physical Inactivity                      31
    Access to Exercise Opportunities         16
    Excessive Drinking                       42
    Alcohol-Impaired Driving Deaths         117
    Sexually Transmitted Infections         154
    Teen Births                              45
    Uninsured                                54
    Primary Care Physicians                 102
    Dentists                                 85
    Mental Health Providers                 175
    Preventable Hospital Stays               72
    Mammography Screening                    58
    Flu Vaccinations                         17
    High School Completion                   73
    Some College                             22
    Unemployment                             77
    Children in Poverty                      68
    Income Inequality                        99
    Children in Single-Parent Households    111
    Social Associations                     105
    Injury Deaths                            86
    Air Pollution - Particulate Matter       23
    Drinking Water Violations                 0
    Severe Housing Problems                 116
    Driving Alone to Work                   143
    Long Commute - Driving Alone              7
    dtype: int64


**Applied Winsorization (Capping at 1st & 99th Percentiles)**


```python
# Cap extreme values at 1st and 99th percentiles
for col in num_vars:
    data[col] = winsorize(data[col], limits=[0.01, 0.01])  # Capping lowest and highest 1%
```

### Computing Summary Statistics


```python
#Numerical columns
print(data.describe())

#prevents NumPy's "MaskedArray" partition warning from being printed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy.lib.function_base")
```

           Premature Death  Adult Smoking  Adult Obesity  Food Environment Index  \
    count      3143.000000    3143.000000    3143.000000             3143.000000   
    mean       9797.942256       0.190248       0.373529                7.553898   
    std        3215.725257       0.039357       0.044539                1.138959   
    min        4150.964198       0.102000       0.241000                3.800000   
    25%        7489.142964       0.164000       0.351000                6.900000   
    50%        9377.864182       0.187000       0.377000                7.700000   
    75%       11671.419865       0.215000       0.403000                8.300000   
    max       20680.182654       0.295000       0.471000                9.500000   
    
           Physical Inactivity  Access to Exercise Opportunities  \
    count          3143.000000                       3143.000000   
    mean              0.266574                          0.616825   
    std               0.051358                          0.228721   
    min               0.154000                          0.014920   
    25%               0.231000                          0.475928   
    50%               0.263000                          0.639460   
    75%               0.301000                          0.788725   
    max               0.399000                          1.000000   
    
           Excessive Drinking  Alcohol-Impaired Driving Deaths  \
    count         3143.000000                      3143.000000   
    mean             0.168638                         0.271258   
    std              0.025798                         0.144177   
    min              0.113531                         0.000000   
    25%              0.150428                         0.181818   
    50%              0.168915                         0.263158   
    75%              0.184923                         0.334499   
    max              0.239718                         0.800000   
    
           Sexually Transmitted Infections  Teen Births  ...  Children in Poverty  \
    count                      3143.000000  3143.000000  ...          3143.000000   
    mean                        379.659306    22.916330  ...             0.193908   
    std                         238.661981    10.888567  ...             0.081765   
    min                           0.000000     3.838262  ...             0.055000   
    25%                         218.250000    15.062000  ...             0.134000   
    50%                         310.900000    21.590599  ...             0.181000   
    75%                         482.500000    29.800138  ...             0.241000   
    max                        1283.100000    54.740275  ...             0.456000   
    
           Income Inequality  Children in Single-Parent Households  \
    count        3143.000000                           3143.000000   
    mean            4.544581                              0.238568   
    std             0.763033                              0.099619   
    min             3.183284                              0.048561   
    25%             4.015987                              0.171399   
    50%             4.433812                              0.221821   
    75%             4.941424                              0.287200   
    max             7.218930                              0.578495   
    
           Social Associations  Injury Deaths  Air Pollution - Particulate Matter  \
    count          3143.000000    3143.000000                         3143.000000   
    mean             11.232336      95.879545                            7.529780   
    std               5.585231      26.709077                            1.660356   
    min               0.000000      43.957459                            3.400000   
    25%               7.874946      77.321739                            6.500000   
    50%              10.769872      93.577201                            7.800000   
    75%              14.124372     110.433502                            8.800000   
    max              29.940120     190.342897                           10.700000   
    
           Drinking Water Violations  Severe Housing Problems  \
    count                3143.000000              3143.000000   
    mean                    0.362711                 0.127399   
    std                     0.480859                 0.038879   
    min                     0.000000                 0.051020   
    25%                     0.000000                 0.101763   
    50%                     0.000000                 0.121837   
    75%                     1.000000                 0.147243   
    max                     1.000000                 0.260707   
    
           Driving Alone to Work  Long Commute - Driving Alone  
    count            3143.000000                   3143.000000  
    mean                0.778497                      0.329965  
    std                 0.067800                      0.125435  
    min                 0.501529                      0.072000  
    25%                 0.750524                      0.236000  
    50%                 0.789653                      0.324000  
    75%                 0.822514                      0.417500  
    max                 0.899578                      0.634000  
    
    [8 rows x 30 columns]



```python
 # Categorical columns
print(data.describe(include='object')) 
```

           statecode countycode fipscode state             county county_clustered
    count       3143       3143     3143  3143               3143             3143
    unique        51        325     3143    51               1878                2
    top           48        001    01001    TX  Washington County                1
    freq         254         49        1   254                 30             3088


## 3. Clustering

I performed **K-Means clustering** for clustering and **PCA** for visualization, selecting the optimal number of clusters (k) based on the Elbow Method and Silhouette Score. The clustering was conducted using premature death rates and various health factors, to identify counties with similar health outcomes.


```python
print(data.columns)
```

    Index(['statecode', 'countycode', 'fipscode', 'state', 'county',
           'county_clustered', 'Premature Death', 'Adult Smoking', 'Adult Obesity',
           'Food Environment Index', 'Physical Inactivity',
           'Access to Exercise Opportunities', 'Excessive Drinking',
           'Alcohol-Impaired Driving Deaths', 'Sexually Transmitted Infections',
           'Teen Births', 'Uninsured', 'Primary Care Physicians', 'Dentists',
           'Mental Health Providers', 'Preventable Hospital Stays',
           'Mammography Screening', 'Flu Vaccinations', 'High School Completion',
           'Some College', 'Unemployment', 'Children in Poverty',
           'Income Inequality', 'Children in Single-Parent Households',
           'Social Associations', 'Injury Deaths',
           'Air Pollution - Particulate Matter', 'Drinking Water Violations',
           'Severe Housing Problems', 'Driving Alone to Work',
           'Long Commute - Driving Alone'],
          dtype='object')



```python
# Selected features for clustering (All health factors)
selected_features = [
    'Adult Smoking', 'Adult Obesity', 'Food Environment Index', 'Physical Inactivity',
    'Access to Exercise Opportunities', 'Excessive Drinking', 'Alcohol-Impaired Driving Deaths',
    'Sexually Transmitted Infections', 'Teen Births', 'Uninsured', 'Primary Care Physicians', 
    'Dentists', 'Mental Health Providers', 'Preventable Hospital Stays', 'Mammography Screening', 
    'Flu Vaccinations', 'High School Completion', 'Some College', 'Unemployment', 
    'Children in Poverty', 'Income Inequality', 'Children in Single-Parent Households', 
    'Social Associations', 'Injury Deaths', 'Air Pollution - Particulate Matter', 
    'Drinking Water Violations', 'Severe Housing Problems', 'Driving Alone to Work', 
    'Long Commute - Driving Alone'
]

# Ensure all selected features exist in the dataset
available_features = [col for col in selected_features if col in data.columns]

# Standardize features before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[selected_features])

# Determine the optimal number of clusters using Elbow Method & Silhouette Score
inertia = []
silhouette_scores = []
cluster_range = range(2, 10)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Clusters (Scaled Data)")
plt.show()
```


    
![png](IAI_HW1_mqlam_files/IAI_HW1_mqlam_29_0.png)
    



```python
# Plot Silhouette Score
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Clusters (Scaled Data)")
plt.show()
```


    
![png](IAI_HW1_mqlam_files/IAI_HW1_mqlam_30_0.png)
    


To determine the optimal number of clusters, I used two methods:

1. **Elbow Method**  
- The elbow point in the inertia plot bends slightly at **k=4 or k=5**
- Beyond k=5, the decrease in inertia slows down, meaning additional clusters do not provide much new information.

2. **Silhouette Score Analysis**  
- The silhouette score is highest at **k=2**, but **k=4** maintains a good balance of separation between clusters.
- The silhouette score drops sharply after k=3, and remains low beyond k=4 or k=5, indicating that clusters become less distinct.

#### Conclusion:
- **k=4 offers the best trade-off** between meaningful structure (Elbow Method) and cluster quality (Silhouette Score).



```python
# Choosing the optimal number of clusters based on the elbow method and silhouette score
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)
```


```python
# Reduce to 2 principal components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA results to DataFrame
data["PCA1"] = X_pca[:, 0]
data["PCA2"] = X_pca[:, 1]

# Scatter plot using PCA components
sns.scatterplot(x=data["PCA1"], y=data["PCA2"], hue=data["Cluster"], palette="viridis")
plt.title("Cluster Visualization using PCA")
plt.show()
```


    
![png](IAI_HW1_mqlam_files/IAI_HW1_mqlam_33_0.png)
    



```python
# Add premature date to the features for calculation
features_to_profile = available_features + ['Premature Death'] if 'Premature Death' in data.columns else available_features

# Compute the mean value of each feature for each cluster
cluster_means = data.groupby("Cluster")[features_to_profile].mean()

# Create a Ranking Table

# For each feature (column), rank the clusters based on the average value.
ranking_table = cluster_means.rank(axis=0, ascending=False)
ranking_table = ranking_table.astype(int)  # Convert to integer rankings for clarity

print("\nRanking Table (each cell shows the rank of the cluster for that feature; 1 = highest average):")
print(ranking_table)
try:
    from IPython.display import display
    display(ranking_table.style.background_gradient(cmap='viridis'))
except ImportError:
    print(ranking_table)
```

    
    Ranking Table (each cell shows the rank of the cluster for that feature; 1 = highest average):
             Adult Smoking  Adult Obesity  Food Environment Index  \
    Cluster                                                         
    0                    3              3                       2   
    1                    1              1                       4   
    2                    2              2                       3   
    3                    4              4                       1   
    
             Physical Inactivity  Access to Exercise Opportunities  \
    Cluster                                                          
    0                          3                                 2   
    1                          1                                 4   
    2                          2                                 3   
    3                          4                                 1   
    
             Excessive Drinking  Alcohol-Impaired Driving Deaths  \
    Cluster                                                        
    0                         2                                2   
    1                         4                                3   
    2                         3                                4   
    3                         1                                1   
    
             Sexually Transmitted Infections  Teen Births  Uninsured  ...  \
    Cluster                                                           ...   
    0                                      4            3          3  ...   
    1                                      1            1          1  ...   
    2                                      3            2          2  ...   
    3                                      2            4          4  ...   
    
             Income Inequality  Children in Single-Parent Households  \
    Cluster                                                            
    0                        4                                     4   
    1                        1                                     1   
    2                        2                                     2   
    3                        3                                     3   
    
             Social Associations  Injury Deaths  \
    Cluster                                       
    0                          1              3   
    1                          4              1   
    2                          2              2   
    3                          3              4   
    
             Air Pollution - Particulate Matter  Drinking Water Violations  \
    Cluster                                                                  
    0                                         4                          4   
    1                                         1                          2   
    2                                         2                          3   
    3                                         3                          1   
    
             Severe Housing Problems  Driving Alone to Work  \
    Cluster                                                   
    0                              4                      3   
    1                              2                      1   
    2                              3                      2   
    3                              1                      4   
    
             Long Commute - Driving Alone  Premature Death  
    Cluster                                                 
    0                                   3                3  
    1                                   2                1  
    2                                   1                2  
    3                                   4                4  
    
    [4 rows x 30 columns]



<style type="text/css">
#T_0f3d7_row0_col0, #T_0f3d7_row0_col1, #T_0f3d7_row0_col3, #T_0f3d7_row0_col8, #T_0f3d7_row0_col9, #T_0f3d7_row0_col13, #T_0f3d7_row0_col19, #T_0f3d7_row0_col23, #T_0f3d7_row0_col27, #T_0f3d7_row0_col28, #T_0f3d7_row0_col29, #T_0f3d7_row1_col6, #T_0f3d7_row2_col2, #T_0f3d7_row2_col4, #T_0f3d7_row2_col5, #T_0f3d7_row2_col7, #T_0f3d7_row2_col10, #T_0f3d7_row2_col11, #T_0f3d7_row2_col12, #T_0f3d7_row2_col14, #T_0f3d7_row2_col15, #T_0f3d7_row2_col16, #T_0f3d7_row2_col17, #T_0f3d7_row2_col25, #T_0f3d7_row2_col26, #T_0f3d7_row3_col18, #T_0f3d7_row3_col20, #T_0f3d7_row3_col21, #T_0f3d7_row3_col22, #T_0f3d7_row3_col24 {
  background-color: #35b779;
  color: #f1f1f1;
}
#T_0f3d7_row0_col2, #T_0f3d7_row0_col4, #T_0f3d7_row0_col5, #T_0f3d7_row0_col6, #T_0f3d7_row0_col10, #T_0f3d7_row0_col11, #T_0f3d7_row0_col15, #T_0f3d7_row0_col16, #T_0f3d7_row0_col17, #T_0f3d7_row1_col12, #T_0f3d7_row1_col25, #T_0f3d7_row1_col26, #T_0f3d7_row1_col28, #T_0f3d7_row2_col0, #T_0f3d7_row2_col1, #T_0f3d7_row2_col3, #T_0f3d7_row2_col8, #T_0f3d7_row2_col9, #T_0f3d7_row2_col13, #T_0f3d7_row2_col18, #T_0f3d7_row2_col19, #T_0f3d7_row2_col20, #T_0f3d7_row2_col21, #T_0f3d7_row2_col22, #T_0f3d7_row2_col23, #T_0f3d7_row2_col24, #T_0f3d7_row2_col27, #T_0f3d7_row2_col29, #T_0f3d7_row3_col7, #T_0f3d7_row3_col14 {
  background-color: #31688e;
  color: #f1f1f1;
}
#T_0f3d7_row0_col7, #T_0f3d7_row0_col12, #T_0f3d7_row0_col18, #T_0f3d7_row0_col20, #T_0f3d7_row0_col21, #T_0f3d7_row0_col24, #T_0f3d7_row0_col25, #T_0f3d7_row0_col26, #T_0f3d7_row1_col2, #T_0f3d7_row1_col4, #T_0f3d7_row1_col5, #T_0f3d7_row1_col10, #T_0f3d7_row1_col11, #T_0f3d7_row1_col14, #T_0f3d7_row1_col15, #T_0f3d7_row1_col16, #T_0f3d7_row1_col17, #T_0f3d7_row1_col22, #T_0f3d7_row2_col6, #T_0f3d7_row3_col0, #T_0f3d7_row3_col1, #T_0f3d7_row3_col3, #T_0f3d7_row3_col8, #T_0f3d7_row3_col9, #T_0f3d7_row3_col13, #T_0f3d7_row3_col19, #T_0f3d7_row3_col23, #T_0f3d7_row3_col27, #T_0f3d7_row3_col28, #T_0f3d7_row3_col29 {
  background-color: #fde725;
  color: #000000;
}
#T_0f3d7_row0_col14, #T_0f3d7_row0_col22, #T_0f3d7_row1_col0, #T_0f3d7_row1_col1, #T_0f3d7_row1_col3, #T_0f3d7_row1_col7, #T_0f3d7_row1_col8, #T_0f3d7_row1_col9, #T_0f3d7_row1_col13, #T_0f3d7_row1_col18, #T_0f3d7_row1_col19, #T_0f3d7_row1_col20, #T_0f3d7_row1_col21, #T_0f3d7_row1_col23, #T_0f3d7_row1_col24, #T_0f3d7_row1_col27, #T_0f3d7_row1_col29, #T_0f3d7_row2_col28, #T_0f3d7_row3_col2, #T_0f3d7_row3_col4, #T_0f3d7_row3_col5, #T_0f3d7_row3_col6, #T_0f3d7_row3_col10, #T_0f3d7_row3_col11, #T_0f3d7_row3_col12, #T_0f3d7_row3_col15, #T_0f3d7_row3_col16, #T_0f3d7_row3_col17, #T_0f3d7_row3_col25, #T_0f3d7_row3_col26 {
  background-color: #440154;
  color: #f1f1f1;
}
</style>
<table id="T_0f3d7">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0f3d7_level0_col0" class="col_heading level0 col0" >Adult Smoking</th>
      <th id="T_0f3d7_level0_col1" class="col_heading level0 col1" >Adult Obesity</th>
      <th id="T_0f3d7_level0_col2" class="col_heading level0 col2" >Food Environment Index</th>
      <th id="T_0f3d7_level0_col3" class="col_heading level0 col3" >Physical Inactivity</th>
      <th id="T_0f3d7_level0_col4" class="col_heading level0 col4" >Access to Exercise Opportunities</th>
      <th id="T_0f3d7_level0_col5" class="col_heading level0 col5" >Excessive Drinking</th>
      <th id="T_0f3d7_level0_col6" class="col_heading level0 col6" >Alcohol-Impaired Driving Deaths</th>
      <th id="T_0f3d7_level0_col7" class="col_heading level0 col7" >Sexually Transmitted Infections</th>
      <th id="T_0f3d7_level0_col8" class="col_heading level0 col8" >Teen Births</th>
      <th id="T_0f3d7_level0_col9" class="col_heading level0 col9" >Uninsured</th>
      <th id="T_0f3d7_level0_col10" class="col_heading level0 col10" >Primary Care Physicians</th>
      <th id="T_0f3d7_level0_col11" class="col_heading level0 col11" >Dentists</th>
      <th id="T_0f3d7_level0_col12" class="col_heading level0 col12" >Mental Health Providers</th>
      <th id="T_0f3d7_level0_col13" class="col_heading level0 col13" >Preventable Hospital Stays</th>
      <th id="T_0f3d7_level0_col14" class="col_heading level0 col14" >Mammography Screening</th>
      <th id="T_0f3d7_level0_col15" class="col_heading level0 col15" >Flu Vaccinations</th>
      <th id="T_0f3d7_level0_col16" class="col_heading level0 col16" >High School Completion</th>
      <th id="T_0f3d7_level0_col17" class="col_heading level0 col17" >Some College</th>
      <th id="T_0f3d7_level0_col18" class="col_heading level0 col18" >Unemployment</th>
      <th id="T_0f3d7_level0_col19" class="col_heading level0 col19" >Children in Poverty</th>
      <th id="T_0f3d7_level0_col20" class="col_heading level0 col20" >Income Inequality</th>
      <th id="T_0f3d7_level0_col21" class="col_heading level0 col21" >Children in Single-Parent Households</th>
      <th id="T_0f3d7_level0_col22" class="col_heading level0 col22" >Social Associations</th>
      <th id="T_0f3d7_level0_col23" class="col_heading level0 col23" >Injury Deaths</th>
      <th id="T_0f3d7_level0_col24" class="col_heading level0 col24" >Air Pollution - Particulate Matter</th>
      <th id="T_0f3d7_level0_col25" class="col_heading level0 col25" >Drinking Water Violations</th>
      <th id="T_0f3d7_level0_col26" class="col_heading level0 col26" >Severe Housing Problems</th>
      <th id="T_0f3d7_level0_col27" class="col_heading level0 col27" >Driving Alone to Work</th>
      <th id="T_0f3d7_level0_col28" class="col_heading level0 col28" >Long Commute - Driving Alone</th>
      <th id="T_0f3d7_level0_col29" class="col_heading level0 col29" >Premature Death</th>
    </tr>
    <tr>
      <th class="index_name level0" >Cluster</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
      <th class="blank col8" >&nbsp;</th>
      <th class="blank col9" >&nbsp;</th>
      <th class="blank col10" >&nbsp;</th>
      <th class="blank col11" >&nbsp;</th>
      <th class="blank col12" >&nbsp;</th>
      <th class="blank col13" >&nbsp;</th>
      <th class="blank col14" >&nbsp;</th>
      <th class="blank col15" >&nbsp;</th>
      <th class="blank col16" >&nbsp;</th>
      <th class="blank col17" >&nbsp;</th>
      <th class="blank col18" >&nbsp;</th>
      <th class="blank col19" >&nbsp;</th>
      <th class="blank col20" >&nbsp;</th>
      <th class="blank col21" >&nbsp;</th>
      <th class="blank col22" >&nbsp;</th>
      <th class="blank col23" >&nbsp;</th>
      <th class="blank col24" >&nbsp;</th>
      <th class="blank col25" >&nbsp;</th>
      <th class="blank col26" >&nbsp;</th>
      <th class="blank col27" >&nbsp;</th>
      <th class="blank col28" >&nbsp;</th>
      <th class="blank col29" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0f3d7_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_0f3d7_row0_col0" class="data row0 col0" >3</td>
      <td id="T_0f3d7_row0_col1" class="data row0 col1" >3</td>
      <td id="T_0f3d7_row0_col2" class="data row0 col2" >2</td>
      <td id="T_0f3d7_row0_col3" class="data row0 col3" >3</td>
      <td id="T_0f3d7_row0_col4" class="data row0 col4" >2</td>
      <td id="T_0f3d7_row0_col5" class="data row0 col5" >2</td>
      <td id="T_0f3d7_row0_col6" class="data row0 col6" >2</td>
      <td id="T_0f3d7_row0_col7" class="data row0 col7" >4</td>
      <td id="T_0f3d7_row0_col8" class="data row0 col8" >3</td>
      <td id="T_0f3d7_row0_col9" class="data row0 col9" >3</td>
      <td id="T_0f3d7_row0_col10" class="data row0 col10" >2</td>
      <td id="T_0f3d7_row0_col11" class="data row0 col11" >2</td>
      <td id="T_0f3d7_row0_col12" class="data row0 col12" >4</td>
      <td id="T_0f3d7_row0_col13" class="data row0 col13" >3</td>
      <td id="T_0f3d7_row0_col14" class="data row0 col14" >1</td>
      <td id="T_0f3d7_row0_col15" class="data row0 col15" >2</td>
      <td id="T_0f3d7_row0_col16" class="data row0 col16" >2</td>
      <td id="T_0f3d7_row0_col17" class="data row0 col17" >2</td>
      <td id="T_0f3d7_row0_col18" class="data row0 col18" >4</td>
      <td id="T_0f3d7_row0_col19" class="data row0 col19" >3</td>
      <td id="T_0f3d7_row0_col20" class="data row0 col20" >4</td>
      <td id="T_0f3d7_row0_col21" class="data row0 col21" >4</td>
      <td id="T_0f3d7_row0_col22" class="data row0 col22" >1</td>
      <td id="T_0f3d7_row0_col23" class="data row0 col23" >3</td>
      <td id="T_0f3d7_row0_col24" class="data row0 col24" >4</td>
      <td id="T_0f3d7_row0_col25" class="data row0 col25" >4</td>
      <td id="T_0f3d7_row0_col26" class="data row0 col26" >4</td>
      <td id="T_0f3d7_row0_col27" class="data row0 col27" >3</td>
      <td id="T_0f3d7_row0_col28" class="data row0 col28" >3</td>
      <td id="T_0f3d7_row0_col29" class="data row0 col29" >3</td>
    </tr>
    <tr>
      <th id="T_0f3d7_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_0f3d7_row1_col0" class="data row1 col0" >1</td>
      <td id="T_0f3d7_row1_col1" class="data row1 col1" >1</td>
      <td id="T_0f3d7_row1_col2" class="data row1 col2" >4</td>
      <td id="T_0f3d7_row1_col3" class="data row1 col3" >1</td>
      <td id="T_0f3d7_row1_col4" class="data row1 col4" >4</td>
      <td id="T_0f3d7_row1_col5" class="data row1 col5" >4</td>
      <td id="T_0f3d7_row1_col6" class="data row1 col6" >3</td>
      <td id="T_0f3d7_row1_col7" class="data row1 col7" >1</td>
      <td id="T_0f3d7_row1_col8" class="data row1 col8" >1</td>
      <td id="T_0f3d7_row1_col9" class="data row1 col9" >1</td>
      <td id="T_0f3d7_row1_col10" class="data row1 col10" >4</td>
      <td id="T_0f3d7_row1_col11" class="data row1 col11" >4</td>
      <td id="T_0f3d7_row1_col12" class="data row1 col12" >2</td>
      <td id="T_0f3d7_row1_col13" class="data row1 col13" >1</td>
      <td id="T_0f3d7_row1_col14" class="data row1 col14" >4</td>
      <td id="T_0f3d7_row1_col15" class="data row1 col15" >4</td>
      <td id="T_0f3d7_row1_col16" class="data row1 col16" >4</td>
      <td id="T_0f3d7_row1_col17" class="data row1 col17" >4</td>
      <td id="T_0f3d7_row1_col18" class="data row1 col18" >1</td>
      <td id="T_0f3d7_row1_col19" class="data row1 col19" >1</td>
      <td id="T_0f3d7_row1_col20" class="data row1 col20" >1</td>
      <td id="T_0f3d7_row1_col21" class="data row1 col21" >1</td>
      <td id="T_0f3d7_row1_col22" class="data row1 col22" >4</td>
      <td id="T_0f3d7_row1_col23" class="data row1 col23" >1</td>
      <td id="T_0f3d7_row1_col24" class="data row1 col24" >1</td>
      <td id="T_0f3d7_row1_col25" class="data row1 col25" >2</td>
      <td id="T_0f3d7_row1_col26" class="data row1 col26" >2</td>
      <td id="T_0f3d7_row1_col27" class="data row1 col27" >1</td>
      <td id="T_0f3d7_row1_col28" class="data row1 col28" >2</td>
      <td id="T_0f3d7_row1_col29" class="data row1 col29" >1</td>
    </tr>
    <tr>
      <th id="T_0f3d7_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_0f3d7_row2_col0" class="data row2 col0" >2</td>
      <td id="T_0f3d7_row2_col1" class="data row2 col1" >2</td>
      <td id="T_0f3d7_row2_col2" class="data row2 col2" >3</td>
      <td id="T_0f3d7_row2_col3" class="data row2 col3" >2</td>
      <td id="T_0f3d7_row2_col4" class="data row2 col4" >3</td>
      <td id="T_0f3d7_row2_col5" class="data row2 col5" >3</td>
      <td id="T_0f3d7_row2_col6" class="data row2 col6" >4</td>
      <td id="T_0f3d7_row2_col7" class="data row2 col7" >3</td>
      <td id="T_0f3d7_row2_col8" class="data row2 col8" >2</td>
      <td id="T_0f3d7_row2_col9" class="data row2 col9" >2</td>
      <td id="T_0f3d7_row2_col10" class="data row2 col10" >3</td>
      <td id="T_0f3d7_row2_col11" class="data row2 col11" >3</td>
      <td id="T_0f3d7_row2_col12" class="data row2 col12" >3</td>
      <td id="T_0f3d7_row2_col13" class="data row2 col13" >2</td>
      <td id="T_0f3d7_row2_col14" class="data row2 col14" >3</td>
      <td id="T_0f3d7_row2_col15" class="data row2 col15" >3</td>
      <td id="T_0f3d7_row2_col16" class="data row2 col16" >3</td>
      <td id="T_0f3d7_row2_col17" class="data row2 col17" >3</td>
      <td id="T_0f3d7_row2_col18" class="data row2 col18" >2</td>
      <td id="T_0f3d7_row2_col19" class="data row2 col19" >2</td>
      <td id="T_0f3d7_row2_col20" class="data row2 col20" >2</td>
      <td id="T_0f3d7_row2_col21" class="data row2 col21" >2</td>
      <td id="T_0f3d7_row2_col22" class="data row2 col22" >2</td>
      <td id="T_0f3d7_row2_col23" class="data row2 col23" >2</td>
      <td id="T_0f3d7_row2_col24" class="data row2 col24" >2</td>
      <td id="T_0f3d7_row2_col25" class="data row2 col25" >3</td>
      <td id="T_0f3d7_row2_col26" class="data row2 col26" >3</td>
      <td id="T_0f3d7_row2_col27" class="data row2 col27" >2</td>
      <td id="T_0f3d7_row2_col28" class="data row2 col28" >1</td>
      <td id="T_0f3d7_row2_col29" class="data row2 col29" >2</td>
    </tr>
    <tr>
      <th id="T_0f3d7_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_0f3d7_row3_col0" class="data row3 col0" >4</td>
      <td id="T_0f3d7_row3_col1" class="data row3 col1" >4</td>
      <td id="T_0f3d7_row3_col2" class="data row3 col2" >1</td>
      <td id="T_0f3d7_row3_col3" class="data row3 col3" >4</td>
      <td id="T_0f3d7_row3_col4" class="data row3 col4" >1</td>
      <td id="T_0f3d7_row3_col5" class="data row3 col5" >1</td>
      <td id="T_0f3d7_row3_col6" class="data row3 col6" >1</td>
      <td id="T_0f3d7_row3_col7" class="data row3 col7" >2</td>
      <td id="T_0f3d7_row3_col8" class="data row3 col8" >4</td>
      <td id="T_0f3d7_row3_col9" class="data row3 col9" >4</td>
      <td id="T_0f3d7_row3_col10" class="data row3 col10" >1</td>
      <td id="T_0f3d7_row3_col11" class="data row3 col11" >1</td>
      <td id="T_0f3d7_row3_col12" class="data row3 col12" >1</td>
      <td id="T_0f3d7_row3_col13" class="data row3 col13" >4</td>
      <td id="T_0f3d7_row3_col14" class="data row3 col14" >2</td>
      <td id="T_0f3d7_row3_col15" class="data row3 col15" >1</td>
      <td id="T_0f3d7_row3_col16" class="data row3 col16" >1</td>
      <td id="T_0f3d7_row3_col17" class="data row3 col17" >1</td>
      <td id="T_0f3d7_row3_col18" class="data row3 col18" >3</td>
      <td id="T_0f3d7_row3_col19" class="data row3 col19" >4</td>
      <td id="T_0f3d7_row3_col20" class="data row3 col20" >3</td>
      <td id="T_0f3d7_row3_col21" class="data row3 col21" >3</td>
      <td id="T_0f3d7_row3_col22" class="data row3 col22" >3</td>
      <td id="T_0f3d7_row3_col23" class="data row3 col23" >4</td>
      <td id="T_0f3d7_row3_col24" class="data row3 col24" >3</td>
      <td id="T_0f3d7_row3_col25" class="data row3 col25" >1</td>
      <td id="T_0f3d7_row3_col26" class="data row3 col26" >1</td>
      <td id="T_0f3d7_row3_col27" class="data row3 col27" >4</td>
      <td id="T_0f3d7_row3_col28" class="data row3 col28" >4</td>
      <td id="T_0f3d7_row3_col29" class="data row3 col29" >4</td>
    </tr>
  </tbody>
</table>




```python
# Premature death of each cluster
if 'Premature Death' in data.columns:
    outcome_summary = data.groupby("Cluster")["Premature Death"].mean()
    print("\nAverage Premature Death by Cluster:")
    print(outcome_summary)
```

    
    Average Premature Death by Cluster:
    Cluster
    0     8025.476419
    1    14299.289175
    2    10984.904852
    3     7409.985244
    Name: Premature Death, dtype: float64


### Analysis: 
#### Cluster 1: Worst Health Outcomes  
Cluster 1 has the highest premature death rate at **14,299 per 100,000 population**. This cluster has the highest levels of **obesity, smoking, and physical inactivity**. Healthcare access is poor, with fewer primary care physicians and dentists. Socioeconomic conditions are also challenging, with **high unemployment, poverty, and income inequality**. Additionally, this cluster has a high number of preventable hospital stays, suggesting a lack of preventive care and routine medical check-ups.

#### Cluster 2: Above-Average Health Outcomes
Cluster 2 has a **premature death rate of 10,984 per 100,000 population**, which is above average but lower than Cluster 1. It also has the **second-highest smoking and obesity rates**. Healthcare access is more limited compared to Clusters 0 and 3, making it harder for people to manage their health. This cluster has **moderate levels of poverty and income inequality**, which may contribute to worse health behaviors and outcomes.

#### Cluster 0: Moderate Health Outcomes  
Cluster 0 has a **premature death rate of 8,025 per 100,000 population**. Health behaviors in this cluster are **moderate**, with mid-range levels of **obesity, smoking, and healthcare access**. Socioeconomic conditions, including **income inequality and poverty**, are also moderate. Compared to Cluster 3, this group has slightly worse health behaviors but better conditions than Clusters 1 and 2.

#### Cluster 3: Best Health Outcomes  
Cluster 3 has the **lowest premature death rate at 7,409 per 100,000 population**. This cluster has **better healthcare access**, with more primary care physicians, dentists, and mental health providers. It also has **lower levels of smoking, obesity, and physical inactivity**, contributing to better health outcomes. **Higher education levels** and **better socioeconomic conditions** help explain the healthier behaviors and lower death rates in this group.

### Summary  
Clusters 1 and 2 have higher premature death rates, worse health behaviors, and more limited healthcare access. Clusters 0 and 3, on the other hand, have better healthcare availability, lower smoking and obesity rates, and stronger socioeconomic conditions. Addressing healthcare gaps and promoting healthier behaviors in Clusters 1 and 2 could help improve overall health outcomes.







```python
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

shapefile_path = '/Users/mai/Desktop/intro ai/hw1/hw1/data/UScounties' 

# Load the U.S. counties shapefile
us_counties = gpd.read_file(shapefile_path)

# Ensure the FIPS codes are strings for merging
us_counties['FIPS'] = us_counties['FIPS'].astype(str)

# Ensure 'fipscode' column is also a string for merging
data['fipscode'] = data['fipscode'].astype(str)

# Remove Alaska (FIPS starts with '02') and Hawaii (FIPS starts with '15')
map_data = us_counties[~us_counties["FIPS"].str.startswith(("02", "15"))]

# Merge the shapefile with clustering data
map_data = map_data.merge(data[['fipscode', 'Cluster']], left_on='FIPS', right_on='fipscode', how='left')

# Convert to Albers Equal Area projection (for better scaling)
map_data = map_data.to_crs(epsg=5070)

# Define the colormap
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=map_data['Cluster'].min(), vmax=map_data['Cluster'].max())

# Create a legend mapping cluster numbers to their colors
legend_patches = [
    mpatches.Patch(color=cmap(norm(cluster)), label=f'Cluster {cluster}')
    for cluster in sorted(map_data['Cluster'].unique())
]

# Plot the clusters
fig, ax = plt.subplots(figsize=(15, 10))
map_data.boundary.plot(ax=ax, linewidth=0.8, color="black")
map_data.plot(column='Cluster', cmap=cmap, ax=ax, legend=True,
              legend_kwds={'label': "Cluster Assignment", 'orientation': "horizontal"})

# Add legend with cluster-color mapping
plt.legend(handles=legend_patches, title="Cluster Colors", loc="lower right", fontsize=10, frameon=True)

# Set title and remove axis labels
plt.title('Clusters of U.S. Counties Based on Health Outcomes (48 States)', fontsize=14)
plt.axis('off')

# Show the map
plt.show()
```


    
![png](IAI_HW1_mqlam_files/IAI_HW1_mqlam_37_0.png)
    


### Analysis
#### **Cluster 1 (Worst Health Outcomes)**
Found mostly in the **Southeast and Appalachian regions**

#### **Cluster 2 (Above-Average Health Risks)**
Scattered across parts of the **Midwest and some urban areas**

#### **Cluster 0 (Moderate Health Outcomes)**
Spread across the **Great Plains and parts of the East Coast**

#### **Cluster 3 (Best Health Outcomes)**
Primarily located in **coastal and urban areas**, including the **Northeast and West Coast**

#### **Conclusion**
The geographic distribution of clusters highlights **significant health disparities across U.S. regions**, emphasizing the need for **targeted healthcare interventions** in high-risk areas like the **Southeast and rural Midwest**.


## 4. Supervised Learning Models

For this analysis, I applied **Linear Regression** and **Random Forest** to predict premature death using various health behavior indicators as input features. These features include factors such as smoking rates, obesity, access to healthcare, physical inactivity, and socioeconomic conditions.

Both models were evaluated based on their **Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score** to determine their accuracy in predicting premature death. 


```python
# Selecting features and target variable for supervised learning
target = 'Premature Death'
X = data[available_features]
y = data[target]

# Split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
# Cross-validation
# Number of folds
cv = 5 

# Linear Regression Cross-Validation
lr_model = LinearRegression()
lr_cv_mse = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='neg_mean_squared_error')
lr_cv_rmse = np.sqrt(-lr_cv_mse)
print("Linear Regression Cross-Validation RMSE scores:", lr_cv_rmse)
print("Mean Linear Regression CV RMSE:", lr_cv_rmse.mean())

# Random Forest Cross-Validation
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cv_mse = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv, scoring='neg_mean_squared_error')
rf_cv_rmse = np.sqrt(-rf_cv_mse)
print("\nRandom Forest Cross-Validation RMSE scores:", rf_cv_rmse)
print("Mean Random Forest CV RMSE:", rf_cv_rmse.mean())
```

    Linear Regression Cross-Validation RMSE scores: [1407.26202558 1434.85937944 1506.12100063 1351.55683354 1402.47478677]
    Mean Linear Regression CV RMSE: 1420.4548051910558
    
    Random Forest Cross-Validation RMSE scores: [1377.3604267  1560.15710771 1478.32063594 1388.73279099 1440.46714989]
    Mean Random Forest CV RMSE: 1449.0076222471284



```python
# Train the final Linear Regression model
lr_model.fit(X_train_scaled, y_train)
# Train the final Random Forest model
rf_model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# Compute performance metrics for Linear Regression
lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, y_pred_lr)

# Compute performance metrics for Random Forest
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_pred_rf)

# Display performance metrics
model_performance = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MSE": [lr_mse, rf_mse],
    "RMSE": [lr_rmse, rf_rmse],
    "R² Score": [lr_r2, rf_r2]
})
print("\nTest Set Performance:")
print(model_performance)
```

    
    Test Set Performance:
                   Model           MSE         RMSE  R² Score
    0  Linear Regression  2.145514e+06  1464.757320  0.800613
    1      Random Forest  2.151298e+06  1466.730542  0.800075



```python
# For Linear Regression: Top 5 features by absolute coefficient value
lr_coef_df = pd.DataFrame({
    'Feature': available_features,
    'Coefficient': lr_model.coef_
})
lr_coef_df['AbsCoefficient'] = lr_coef_df['Coefficient'].abs()
top5_lr = lr_coef_df.sort_values(by='AbsCoefficient', ascending=False).head(5)
print("\nTop 5 features influencing premature death (Linear Regression):")
print(top5_lr[['Feature', 'Coefficient']])
```

    
    Top 5 features influencing premature death (Linear Regression):
                                Feature  Coefficient
    23                    Injury Deaths  1243.500789
    0                     Adult Smoking   659.499877
    19              Children in Poverty   490.289116
    7   Sexually Transmitted Infections   468.513797
    8                       Teen Births   438.457214



```python
# For Random Forest: Top 5 features by feature importance
rf_importance_df = pd.DataFrame({
    'Feature': available_features,
    'Importance': rf_model.feature_importances_
})
top5_rf = rf_importance_df.sort_values(by='Importance', ascending=False).head(5)
print("\nTop 5 features influencing premature death (Random Forest):")
print(top5_rf)
```

    
    Top 5 features influencing premature death (Random Forest):
                    Feature  Importance
    19  Children in Poverty    0.448671
    23        Injury Deaths    0.192535
    8           Teen Births    0.084435
    5    Excessive Drinking    0.030008
    3   Physical Inactivity    0.025517


#### **Comparison of 2 models:**
The performance difference is extremely small, but Linear Regression performs slightly better in all three metrics (lower MSE, lower RMSE, and higher R²). Thus, Linear Regression is preferable because it provides better interpretability, making it easier to understand the impact of each predictor on premature death. Random Forest is typically more robust for capturing non-linear relationships, but in this case, it does not offer a significant advantage.



## 5. Recommendations to reduce premature death

#### Cluster of Allegheny County


```python
allegheny_cluster = data[data['county'].str.contains("Allegheny", case=False, na=False)]
print(allegheny_cluster[['county', 'state', 'Cluster']])
```

                    county state  Cluster
    2286  Allegheny County    PA        3


Since Allegheny County belongs to Cluster 3, which has the best health outcomes, the focus should be on maintaining progress, addressing remaining disparities, and enhancing preventive care. In addition, based on the top 5 features identified by the Linear Regression and Random Forest models, key areas for intervention include **injury prevention, reducing smoking, addressing child poverty, improving sexual health, and lowering teen birth rates.**

#### Immediate Strategies

**Expand Preventive Care Programs:** Encourage regular health screenings, vaccinations, and chronic disease management to reduce preventable hospital visits and improve long-term health outcomes.

**Strengthen Injury Prevention:** Enforce stricter DUI laws and road safety improvements and provide workplace safety training in high-risk industries.

**Enhance Smoking Cessation:** Strengthen anti-smoking campaigns, targeting youth and low-income populations.

**Improve Sexual & Reproductive Health:** Increase access to contraception, STI screenings, and education in clinics and schools. Expand teen pregnancy prevention programs in at-risk communities.

**Enhance Workplace Wellness Initiatives:** Encourage employers to offer wellness programs, such as gym memberships, mental health support, and nutrition counseling, to keep the workforce healthy.

#### Long-term Strategies

**Maintain Strong Healthcare Access:** Allegheny County already has a high number of primary care physicians, dentists, and mental health providers. Efforts should focus on ensuring equitable access so all residents, including lower-income populations, can benefit.

**Promote Health Equity:** Even in well-performing counties, disparities exist. Target low-income neighborhoods and underserved populations with programs that provide affordable healthcare, healthy food access, and fitness opportunities.

**Invest in Data-Driven Health Improvements:** Use county-level health data to target specific neighborhoods with higher smoking, obesity, and substance use rates and tailor intervention programs.

**Reduce Socioeconomic Barriers** Strengthen childhood poverty reduction programs like affordable childcare and early education.

## 6. External Libraries and References

- **Scikit-learn Developers.** (n.d.). *SimpleImputer (sklearn.impute.SimpleImputer)*. Retrieved from [https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

- **SciPy Developers.** (n.d.). *Winsorization (scipy.stats.mstats.winsorize)*. Retrieved from [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html)

- **Analytics Vidhya.** (2021). *K-Means: Getting the Optimal Number of Clusters*. Retrieved from [https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters](https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters)

- **IPython Developers.** (n.d.). *IPython Display Module*. Retrieved from [https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html)

- **J. Cutrer.** (n.d.). *Learn GeoPandas: Plotting US Maps in Python*. Retrieved from [https://jcutrer.com/python/learn-geopandas-plotting-usmaps](https://jcutrer.com/python/learn-geopandas-plotting-usmaps)

- **Ruiz, J. L.** (n.d.). *Plot Maps from the US Census Bureau Using GeoPandas and Contextily in Python*. Retrieved from [https://medium.com/@jl_ruiz/plot-maps-from-the-us-census-bureau-using-geopandas-and-contextily-in-python-df787647ef77](https://medium.com/@jl_ruiz/plot-maps-from-the-us-census-bureau-using-geopandas-and-contextily-in-python-df787647ef77)

- **Scikit-learn Developers.** (n.d.). *Forest of trees-based ensemble methods (Random Forest)*. In *scikit-learn: Machine Learning in Python*. Retrieved from [https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py)

- **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E.** (2011). *Scikit-learn: Machine Learning in Python.* *Journal of Machine Learning Research*, 12, 2825–2830. Retrieved from [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)



```python

```


```python

```

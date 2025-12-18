# Data Pre-processing
Source: [DataCamp Article](https://www.datacamp.com/blog/data-preprocessing)

***
## Data Pre-processing
Data Preprocessing is the process of converting raw data into a format that can be processed more efficiently and accurately in tasks such as:
1. Data Analysis
2. Machine Learning
3. Data Science
4. AI

## Steps in Data Pre-processing

### Data Cleaning
It is the process of identifying and correcting errors or inconsistencies in the data to ensure it is accurate and complete. The objective is to address issues that can distort analysis or model performance. 
For example:
1. Handling missing values: these rows can be removed or even filled with predictive models.
2. Removing duplicates: Duplication must be eliminated.
3. Correcting inconsistent formats: Standardizing formats - data formats, string cases etc.

Common Techniques:
1. Imputation: This includes filling the missing data with either a caclulated estimate (mean, median or mode) or includes predictive modelling where prediction models fill the data.
2. Deletion: Removing the rows and columns with missing values.
3. Outlier Removal: This includes removing values that siginificantly deviate from the rest of the data. One approach is through **Z-Score method** which involves flagging and removing data significantly distant from the mean. The threshold is explicitly set. Other techniques include visualization through graphs like scatter plot that flag the anomalies.

### Data Integration
Data integration involves combining data from multiple sources to create a unified dataset. This is often important when data is collected from different source systems.

### Data Transformation
Data transformation icnludes converting data into suitable formats for analysis, ML, or mining.
For example:
1. Scaling and Normalization: Adjusting numeric values to a common scale.
2. Encoding categorical variables: Converting categorical data into numerical values using one-hot or label encoding techniques.
3. Feature engineering and extraction: Creating new features or selecting important ones to improve the model's performance.

Common Techniques:
1. **Data Encoding**: Data Encoding involves conversion of categorical data into numering representations that ML models can understand. Moreover, the numerical representation must be as such that it does not cause the model to hallucinate. 
To do categorical encoding, we have a few techniques:
    a. Label encoding: If there are sensor IDs (A, B, C) you cannot bear to convert them to merely 1, 2 and 3 as then the model will associate weights to the ID as well. It will hallucinate thinking C (3) is more distant than A (1).
    Therefore, assigning just unique numerical representation causes unintended ordinal relationship between them.
    b. One-hot encoding: It creates binary column for each category:
    ```python
        Sensor ID: A, B, C
        A → [1, 0, 0]
        B → [0, 1, 0]
        C → [0, 0, 1]
    ```
    c. Ordinal encoding: When we want such ordinal relationship. Each category is mapped to a corresponding integer value that reflects its ranking.

2. **Scaling**: It ensures that the columns are on a similar scale. For instance, If one feature ranges to 100 000 and another to 1, the large one dominates distance. The model becomes a single-feature model.
Scaling ensures that each feature becomes comparable in terms of magnitude while keeping their relative structure. A common method is Standardization doing using `StandardScaler`.

3. **Data Augmentation**: It involves creation of synthetic data in order to increase the data size. It is helpful in the cases of images where a large data is required for effectiveness.

### Data Reduction
Data Reduction simplifies the dataset by reducing the number of features while preserving the essential information.
Techniques to do this include: Feature selection, Principle component analysis, Sampling methods

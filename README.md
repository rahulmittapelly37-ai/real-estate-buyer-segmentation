#  Real Estate Buyer Segmentation & Investment Profiling (ML)

##  Project Overview
This project applies **machine learning clustering** to segment real estate buyers and profile their investment behavior. By analyzing demographics, financing patterns, and acquisition purposes, the model helps real estate companies improve **market intelligence**, optimize **marketing spend**, and target **profitable buyer segments**.

## Objectives
- Discover hidden buyer patterns using clustering.
- Segment clients by investment purpose, demographics, and loan behavior.
- Provide actionable insights for marketing and property recommendations.
- Deploy an interactive **Streamlit dashboard** for visualization.

## Methodology
1. **Data Cleaning**
   - Handle missing attributes.
   - Normalize categorical labels.
   - Remove duplicate client entries.
2. **Feature Encoding**
   - Apply Label Encoding and One‑Hot Encoding for categorical variables:
     - `client_type`, `region`, `acquisition_purpose`, `referral_channel`, `country`
3. **Feature Scaling**
   - Normalize numeric variables (`age`, `satisfaction_score`) using StandardScaler or MinMaxScaler.
4. **Clustering Model Selection**
   - **K‑Means**: Efficient, easy to interpret.
   - **Hierarchical Clustering**: Reveals nested relationships, validates K‑Means.
5. **Optimal Cluster Selection**
   - **Elbow Method**: Identifies optimal number of clusters.
   - **Silhouette Score**: Measures clustering quality.
6. **Cluster Interpretation**
   - Analyze clusters by investment purpose, geographic distribution, loan behavior, and demographics.

## Recommended Buyer Segments
- **C1 – Global Investors**: High income, investment purchases.  
- **C2 – First‑Time Buyers**: Younger, loan dependent.  
- **C3 – Corporate Buyers**: Companies purchasing multiple units.  
- **C4 – Luxury Investors**: High satisfaction, large investments.  

##  Streamlit Dashboard Features
- **Buyer Segmentation Overview**: Cluster distribution.  
- **Investor Behavior Dashboard**: Investment patterns by cluster.  
- **Geographic Buyer Analysis**: Regional segmentation maps.  
- **Segment Insights Panel**: Descriptive statistics per cluster.  
- **User Controls**: Filter by country, region, acquisition purpose, client type.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Streamlit Setup ---
st.set_page_config(page_title="Climate Cluster Dashboard", layout="wide")

st.title("🌍 Climate Change & Agriculture: Clustering Dashboard")


df = pd.read_csv("climate_change_impact_on_agriculture_2024.csv")
st.subheader("Raw Data")
st.dataframe(df.head())

# --- Data Preprocessing ---
st.subheader("⚙️ Data Preprocessing")

df_clean = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Remove outliers
for col in numerical_columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

# Label Encoding
X_processed = df_clean.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col])
    label_encoders[col] = le

# Scaling
scaler = StandardScaler()
X_processed[numerical_columns] = scaler.fit_transform(X_processed[numerical_columns])

# Feature Selection (You can change indices based on your project)
X = X_processed.iloc[:, [7, 14]]

# --- GMM Clustering ---
st.subheader("🔍 GMM Clustering")
gmm = joblib.load("gmm.pkl")
df_clean['Cluster'] = gmm.fit_predict(X)

st.success("✅ GMM Clustering applied successfully!")

# --- Predict New Cluster (Sidebar Input) ---
with st.sidebar.form("predict_form"):
    st.subheader("🧠 Predict Cluster")
    st.markdown("Enter new data below:")

    user_input = {}
    cols = ['Crop_Yield_MT_per_HA', 'Economic_Impact_Million_USD']
    for col in cols:
        user_input[col] = st.number_input(f"{col}", value=0.0, key=col)

    submitted = st.form_submit_button("Predict Cluster")

if submitted:
    input_df = pd.DataFrame([user_input])
    scalerInput = StandardScaler()
    input_df[cols] = scalerInput.fit_transform(input_df[cols])
    prediction = gmm.predict(input_df)[0]
    st.sidebar.success(f"🌟 Predicted Cluster: {prediction}")


# --- Insights Section ---
st.subheader("📊 Cluster Insights")

result = df_clean.copy()
# 1. Cluster count
Cluster_counts = result['Cluster'].value_counts().sort_index()
st.write("### Number of points in each Cluster")
st.bar_chart(Cluster_counts)

# 2. Cluster by Region
if 'Region' in df_clean.columns:
    st.write("### Number of points in each Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(result.groupby(['Region', 'Cluster']).size().unstack().fillna(0), annot=True, fmt='g', cmap='Blues', ax=ax)
    st.pyplot(fig)

# 3. Top Countries by Cluster
if 'Country' in df_clean.columns:
    country_Cluster = result.groupby(['Country', 'Cluster']).size().unstack().fillna(0)
    top_countries = country_Cluster.sum(axis=1).sort_values(ascending=False).head(10).index
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(country_Cluster.loc[top_countries], annot=True, fmt='g', cmap='Blues', ax=ax)
    st.pyplot(fig)

# 4. Yearly trend
if 'Year' in df_clean.columns:
    yearly_trend = result.groupby(['Year', 'Cluster']).size().unstack().fillna(0)
    yearly_trend_reset = yearly_trend.reset_index().melt(id_vars='Year', var_name='Cluster', value_name='Count')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=yearly_trend_reset, x='Year', y='Count', hue='Cluster', marker='o')
    plt.title('Cluster Distribution Over Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    plt.grid(True)
    st.pyplot(fig)
    
st.subheader("🗓️ CO₂ vs Temp Trends by Time Bin")
try:
    result['Year_Bin'] = pd.cut(result['Year'], bins=[1995, 2005, 2015, 2025], labels=['1996-2005', '2006-2015', '2016-2025'])
    g = sns.FacetGrid(result, col='Year_Bin', hue='Cluster', palette='Set2', height=4, aspect=1.2)
    g.map_dataframe(sns.scatterplot, x='Average_Temperature_C', y='CO2_Emissions_MT', alpha=0.6)
    g.add_legend()
    g.set_axis_labels("Average Temperature (°C)", "CO₂ Emissions (MT)")
    g.fig.subplots_adjust(top=0.8)
    g.fig.suptitle("CO₂ Emissions vs Temperature by Cluster across Time Bins")
    st.pyplot(g.fig)
except:
    st.warning("Could not plot time-binned trend. Check if `Year`, `Average_Temperature_C`, and `CO2_Emissions_MT` exist.")

st.subheader("🧭 Dominant Cluster per Region Over Time")
try:
    region_year_Cluster = result.groupby(['Region', 'Year'])['Cluster'].apply(lambda x: x.mode()[0])
    dominant_Cluster = region_year_Cluster.unstack().fillna(-1)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(dominant_Cluster, annot=True, fmt='.0f', cmap='viridis', cbar_kws={'label': 'Cluster'})
    plt.title('Dominant Cluster per Region Over Time')
    plt.xlabel('Year')
    plt.ylabel('Region')
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not compute dominant Clusters by region.: {e}" )

st.subheader("📍 Cluster Distribution by Country and Region")
country_name = st.selectbox("Select a country", sorted(result['Country'].unique()), index=0)
country_Cluster = result[result['Country'] == country_name][['Year', 'Region', 'Cluster']].sort_values('Year')

Cluster_group = country_Cluster.groupby(['Year', 'Region', 'Cluster']).size().reset_index(name='count')
unique_Clusters = Cluster_group['Cluster'].unique()

for cl in unique_Clusters:
    Cluster_data = Cluster_group[Cluster_group['Cluster'] == cl]
    pivot = Cluster_data.pivot(index='Region', columns='Year', values='count').fillna(0)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt='g', cmap='YlGnBu', ax=ax)
    ax.set_title(f"Heatmap for Cluster {cl} in {country_name}")
    st.pyplot(fig)

st.subheader("📈 Temporal Cluster Peaks")
yearly_trend = result.groupby(['Year', 'Cluster']).size().unstack(fill_value=0)
for Cluster_id in sorted(result['Cluster'].unique()):
    dominant_years = yearly_trend[yearly_trend[Cluster_id] == yearly_trend[Cluster_id].max()].index.tolist()
    st.markdown(f"- **Cluster {Cluster_id}** peaked in: {', '.join(map(str, dominant_years))}")

st.subheader("📐 Feature Mean & Standard Deviation by Cluster")
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
if 'Cluster' not in numeric_columns:
    numeric_columns.append('Cluster')

# Compute mean and std
mean_result = result[numeric_columns].groupby('Cluster').mean().T
std_result = result[numeric_columns].groupby('Cluster').std().T

# Display side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Mean by Cluster")
    st.dataframe(mean_result)

with col2:
    st.markdown("### 📈 Standard Deviation by Cluster")
    st.dataframe(std_result)


st.subheader("📊 Categorical Breakdown by Cluster")

clus_0 = result[result['Cluster'] == 0]
clus_1 = result[result['Cluster'] == 1]

def pie_chart(df, col, axes, Clusterno):
    s = f"{Clusterno} {col}"
    cnts = df[col].value_counts()
    axes.pie(cnts, labels=cnts.index, autopct='%.1f%%', startangle=90, colors=sns.color_palette('colorblind'))
    axes.set_title(s.upper(), fontsize=14, fontweight='bold')

def bar_plt(df, col, axes, Clusterno):
    s = f"{Clusterno} {col}"
    cnts = df[col].value_counts()
    bars = axes.bar(cnts.index, cnts.values)
    axes.bar_label(bars)
    axes.set_title(s.upper(), fontsize=14, fontweight='bold')
    axes.get_yaxis().set_visible(False)
    axes.set_xlabel(col)

# Country & Region
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
bar_plt(clus_0, 'Country', axes[0,0], 'Cluster 0')
bar_plt(clus_1, 'Country', axes[0,1], 'Cluster 1')
pie_chart(clus_0, 'Region', axes[1,0], 'Cluster 0')
pie_chart(clus_1, 'Region', axes[1,1], 'Cluster 1')
st.pyplot(fig)

# Crop Type
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
bar_plt(clus_0, 'Crop_Type', axes[0], 'Cluster 0')
bar_plt(clus_1, 'Crop_Type', axes[1], 'Cluster 1')
st.pyplot(fig)

# Adaptation Strategies
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
pie_chart(clus_0, 'Adaptation_Strategies', axes[0], 'Cluster 0')
pie_chart(clus_1, 'Adaptation_Strategies', axes[1], 'Cluster 1')
st.pyplot(fig)








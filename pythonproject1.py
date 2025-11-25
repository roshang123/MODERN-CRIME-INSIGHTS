### EDA ###
#---------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

df=pd.read_csv(r"C:\Users\HP\Downloads\Crime_Data_from_2020_to_Present.csv")
print(df.columns)                  # Column names
print(df.shape)                    # Number of rows and columns
print(df.info())                   # Data types and non-null counts
print(df.describe)                 # Summary for numerical columns
print(df.head())                   # First five records
print(df.tail())                   # Last five records
print(df.isnull().sum())           # Total missing values per column
print(df.duplicated().sum())       # Check for duplicate rows
print(df.dropna())                 # Remove missing/duplicate values
print(df.fillna(method='ffill'))   # Fill missing/duplicate values

### Objective 1 ###
#Analyze Crime Trends Over Time 
#To identity and visualize now cifferent types of sanes leg volent, properly, cyberdime) have changed from 2020 to the present  
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

# Convert the date column to datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df = df.dropna(subset=['DATE OCC'])

# Extract year and month
df['Year'] = df['DATE OCC'].dt.year
df['Month'] = df['DATE OCC'].dt.to_period('M')

# =========================
# 1. Total Crimes by Year
# =========================
yearly_counts = df.groupby('Year').size().reset_index(name='Total Crimes')
fig1 = px.bar(
    yearly_counts,
    x='Year',
    y='Total Crimes',
    title='Total Crimes per Year',
    text='Total Crimes',
    labels={'Total Crimes': 'Number of Crimes'}
)
fig1.show()

# ================================
# 2. Line Chart of Top Crime Types
# ================================
monthly_crimes = df.groupby(['Month', 'Crm Cd Desc']).size().reset_index(name='Count')
monthly_crimes['Month'] = monthly_crimes['Month'].dt.to_timestamp()

top_crimes = monthly_crimes.groupby('Crm Cd Desc')['Count'].sum().sort_values(ascending=False).head(5).index
filtered_crimes = monthly_crimes[monthly_crimes['Crm Cd Desc'].isin(top_crimes)]

fig2 = px.line(
    filtered_crimes,
    x='Month',
    y='Count',
    color='Crm Cd Desc',
    title='Top 5 Crime Trends Over Time',
    labels={'Month': 'Date', 'Count': 'Crime Count', 'Crm Cd Desc': 'Crime Type'}
)
fig2.update_layout(xaxis=dict(dtick="M3", tickformat="%b\n%Y"))
fig2.show()

# ===============================
# 3. Stacked Area Chart of Crimes
# ===============================
fig3 = px.area(
    filtered_crimes,
    x='Month',
    y='Count',
    color='Crm Cd Desc',
    title='Stacked Area Chart of Top 5 Crime Types',
    labels={'Month': 'Date', 'Count': 'Crime Count', 'Crm Cd Desc': 'Crime Type'}
)
fig3.update_layout(xaxis=dict(dtick="M3", tickformat="%b\n%Y"))
fig3.show()

# ================================
# 4. Pie Chart of Overall Crime Types
# ================================
overall_counts = df['Crm Cd Desc'].value_counts().head(10).reset_index()
overall_counts.columns = ['Crime Type', 'Count']

fig4 = px.pie(
    overall_counts,
    values='Count',
    names='Crime Type',
    title='Top 10 Crime Types (Overall Distribution)',
    hole=0.3
)
fig4.show()



### Objective 2 ###
# Detect High Crime Areas (Hotspots ) 
#To use geospatial analysis to locate and monitor areas with hiogh crime rates and understand the  
#sacio- economic factors behind them  give me code for this objective 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

# Convert date to datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')

# Optional: Filter last 12 months
recent_df = df[df['DATE OCC'] >= pd.Timestamp.now() - pd.DateOffset(months=12)]

# Extract and clean latitude and longitude
if 'LAT' in recent_df.columns and 'LON' in recent_df.columns:
    geo_df = recent_df[['LAT', 'LON']].dropna()
else:
    geo_df = recent_df['Location'].str.extract(r'\((.*), (.*)\)').astype(float).dropna()
    geo_df.columns = ['LAT', 'LON']

# ==========================
# 1. KDE Heatmap (Hotspots)
# ==========================
plt.figure(figsize=(10, 8))
sns.kdeplot(
    data=geo_df,
    x='LON',
    y='LAT',
    fill=True,
    cmap='Reds',
    bw_adjust=0.5,
    thresh=0.05
)
plt.title('Crime Hotspot Density (KDE Heatmap)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# ==========================
# 2. Hexbin Plot (Clusters)
# ==========================
plt.figure(figsize=(10, 8))
plt.hexbin(
    x=geo_df['LON'],
    y=geo_df['LAT'],
    gridsize=60,
    cmap='inferno',
    bins='log'
)
plt.colorbar(label='Crime Frequency (log scale)')
plt.title('Crime Clusters (Hexbin Map)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()




### Objective 3 ###
##Predict Future Crime Patterns Using Machine Leaming 
#to build predictive models that can help forecast potential crime trends or
#spikes based on historical data  and external factors (e.g.unemployment, population density) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

# Clean and convert date
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df = df.dropna(subset=['DATE OCC'])

# Extract month and year
df['YearMonth'] = df['DATE OCC'].dt.to_period('M')
monthly_crime = df.groupby('YearMonth').size().reset_index(name='Crime_Count')
monthly_crime['YearMonth'] = monthly_crime['YearMonth'].astype(str)

# Encode time as numbers for regression (0, 1, 2, ...)
monthly_crime['MonthIndex'] = np.arange(len(monthly_crime))

# ===========================
# Linear Regression (NumPy)
# ===========================
X = monthly_crime['MonthIndex'].values
y = monthly_crime['Crime_Count'].values

# Fit linear regression using polyfit
slope, intercept = np.polyfit(X, y, 1)
monthly_crime['Predicted'] = slope * X + intercept

# Predict for next 6 months
future_months = 6
future_index = np.arange(len(X), len(X) + future_months)
future_pred = slope * future_index + intercept

# Create a future DataFrame
future_dates = pd.date_range(start=monthly_crime['YearMonth'].iloc[-1], periods=future_months+1, freq='M')[1:]
future_df = pd.DataFrame({
    'YearMonth': future_dates.strftime('%Y-%m'),
    'MonthIndex': future_index,
    'Predicted': future_pred
})

# ============================
# 📊 Plot 1: Trends + Forecast
# ============================
plt.figure(figsize=(12, 6))
plt.plot(monthly_crime['YearMonth'], y, label='Actual', marker='o')
plt.plot(monthly_crime['YearMonth'], monthly_crime['Predicted'], label='Fitted Line', linestyle='--')
plt.plot(future_df['YearMonth'], future_df['Predicted'], label='Future Forecast', color='red', linestyle='dotted', marker='o')
plt.xticks(rotation=45)
plt.title('Crime Trend Forecast Using Linear Regression')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 📊 Plot 2: Residual Plot
# ============================
residuals = y - monthly_crime['Predicted']

plt.figure(figsize=(10, 5))
sns.residplot(x=monthly_crime['MonthIndex'], y=residuals, lowess=True, color='purple')
plt.title('Residual Plot: Prediction Errors Over Time')
plt.xlabel('Month Index')
plt.ylabel('Residual (Actual - Predicted)')
plt.grid(True)
plt.tight_layout()
plt.show()


### Objective 4 ###
#Compare Crime Rates Across Regions or Cities  
#To conduct comparative analysis between multiple regions or sites to identify which areas are safer
# or more vulnerable and why.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

# Check and clean date
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df = df.dropna(subset=['DATE OCC'])

# Use 'AREA NAME' if available, or 'AREA', or 'Zip Code'
area_col = 'AREA NAME' if 'AREA NAME' in df.columns else 'AREA'

# ================================
# 📊 Plot 1: Total Crime by Area
# ================================
area_counts = df[area_col].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=area_counts.values, y=area_counts.index, palette='rocket')
plt.title('Top 10 Areas with Highest Crime Count')
plt.xlabel('Number of Crimes')
plt.ylabel('Region')
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================
# 📊 Plot 2: Monthly Crime Boxplot by Area
# ================================

# Extract year-month
df['YearMonth'] = df['DATE OCC'].dt.to_period('M')

# Group by area and month
monthly_area_crime = df.groupby([area_col, 'YearMonth']).size().reset_index(name='Crime_Count')

# Filter top 5 regions for clearer boxplot
top_areas = area_counts.index[:5]
filtered = monthly_area_crime[monthly_area_crime[area_col].isin(top_areas)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered, x=area_col, y='Crime_Count', palette='Set2')
plt.title('Monthly Crime Distribution by Area (Top 5)')
plt.xlabel('Area')
plt.ylabel('Monthly Crime Count')
plt.grid(True)
plt.tight_layout()
plt.show()




### Objective 5 ###
# Identify Seasonal or Temporal Crime Patterns  
#To detect patterns in crime data based on time of day ,day of the week,
# or season, helping to understand when crimes are more likely to occur
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

# Parse datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df = df.dropna(subset=['DATE OCC'])

# ============
# TIME OF DAY
# ============

# Parse time from TIME OCC (some may be like 1300, 0730)
df['Hour'] = pd.to_datetime(df['TIME OCC'].astype(str).str.zfill(4), format='%H%M', errors='coerce').dt.hour

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Hour', palette='magma')
plt.title('Crime Count by Hour of the Day')
plt.xlabel('Hour (24h format)')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()

# =================
# DAY OF THE WEEK
# =================

df['DayOfWeek'] = df['DATE OCC'].dt.day_name()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='DayOfWeek', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], palette='coolwarm')
plt.title('Crime Count by Day of the Week')
plt.xlabel('Day')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================
# OPTIONAL: SEASONAL PATTERN
# ======================

# Function to map season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['DATE OCC'].dt.month.apply(get_season)

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Season', order=['Winter', 'Spring', 'Summer', 'Fall'], palette='Set3')
plt.title('Crime Count by Season')
plt.xlabel('Season')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()


from utils import db_connect
from utils import load_or_download_data

df = load_or_download_data()
engine = db_connect()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression, SelectKBest

# Load raw data
df = pd.read_csv("data/raw/AB_NYC_2019.csv")

# Clean + filter data
df.dropna(subset=['reviews_per_month'], inplace=True)
df.drop(columns=['id', 'name', 'host_id', 'host_name', 'latitude', 'longitude', 'last_review'], inplace=True)

# Drop duplicates
df = df.drop_duplicates(subset=df.columns.difference(['id']))

# Create engineered features
df['reviews_per_year'] = df['reviews_per_month'] * 12
df['host_type'] = df['calculated_host_listings_count'].apply(lambda x: "solo" if x == 1 else "multi")
df['host_type_factor'] = df['host_type'].map({'solo': 0, 'multi': 1})
df['neigh_group'] = pd.factorize(df['neighbourhood_group'])[0]
df['room_type_factor'] = pd.factorize(df['room_type'])[0]

# Remove outliers
mn_iqr = df['minimum_nights'].quantile(0.75) - df['minimum_nights'].quantile(0.25)
up_mn = df['minimum_nights'].quantile(0.75) + 1.5 * mn_iqr
low_mn = df['minimum_nights'].quantile(0.25) - 1.5 * mn_iqr

pr_iqr = df['price'].quantile(0.75) - df['price'].quantile(0.25)
up_price = df['price'].quantile(0.75) + 1.5 * pr_iqr

df = df[(df['minimum_nights'] >= low_mn) & (df['minimum_nights'] <= up_mn)]
df = df[df['price'] <= up_price]

# Select features
features = [
    "minimum_nights", "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365", "neigh_group",
    "reviews_per_year", "room_type_factor", "host_type_factor"
]

X = df[features]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Feature selection
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X_train_scaled, y_train)

selected_cols = X_train_scaled.columns[selector.get_support()]
X_train_sel = pd.DataFrame(selector.transform(X_train_scaled), columns=selected_cols)
X_test_sel = pd.DataFrame(selector.transform(X_test_scaled), columns=selected_cols)

# Merge back price for future modeling/visualizing
X_train_sel['price'] = y_train.values
X_test_sel['price'] = y_test.values

# Save processed data
X_train_sel.to_csv("data/processed/train_data.csv", index=False)
X_test_sel.to_csv("data/processed/test_data.csv", index=False)

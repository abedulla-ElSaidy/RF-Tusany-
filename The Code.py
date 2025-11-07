#  Drought Prediction and Spatial Visualization — *Bruna River Generalization Example*  
**Author:** Elsaidy, Abedulla (2025)  
**Last Updated:** November 2025  
# RF
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

TRAIN_FILE = r"Train- without Nan.csv"
TEST_FILE  = r"Test- without Nan.csv"

TARGET_COLUMN = 'Drought'
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.85
CATEGORICAL_COLUMNS = ['Provincia', 'Codice']

RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}

SMOTE_RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 2  # optimized for extreme imbalance

SPEI_FEATURES = ['SPEI01', 'SPEI03', 'SPEI06', 'SPEI09', 'SPEI12']
SPI_FEATURES  = ['SPI01', 'SPI03', 'SPI06', 'SPI09', 'SPI12']
HYDR_FEATURES = ['gr', 'if', 'rf', 'ws']
WEATHER_FEATURES = ['ae', 'pr', 'td', 'tp']
GEO_FEATURES  = ['Lat', 'Lon', 'Quota [m]']

def generate_features(df):
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLUMNS if c in df.columns], drop_first=True)
    
    if 'Month' in df.columns:
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(1).astype(int)
        df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.fillna(0)

def identify_decorrelated_features(training_data, target_col, corr_threshold):
    cols_to_drop_init = [target_col, 'Year', 'Month', 'E', 'N', 'sin_month', 'cos_month'] + CATEGORICAL_COLUMNS
    X_temp = training_data.drop(columns=[c for c in cols_to_drop_init if c in training_data.columns], errors='ignore')
    
    oh_cols = [c for c in X_temp.columns if any(cat_col in c for cat_col in CATEGORICAL_COLUMNS)]
    X_temp_num = X_temp.drop(columns=oh_cols, errors='ignore')
    
    corr_matrix = X_temp_num.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_num = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    
    decorrelated_features = [c for c in X_temp_num.columns if c not in to_drop_num]
    final_decorrelated = decorrelated_features + oh_cols
    if 'sin_month' in training_data.columns:
        final_decorrelated += ['sin_month', 'cos_month']

    return [c for c in final_decorrelated if c in training_data.columns]

def get_base_feature_sets(training_data):
    decorrelated_features = identify_decorrelated_features(training_data, TARGET_COLUMN, CORRELATION_THRESHOLD)
    oh_cols = [c for c in training_data.columns if any(cat_col in c for cat_col in CATEGORICAL_COLUMNS)]
    all_weather_hydr = WEATHER_FEATURES + HYDR_FEATURES + oh_cols + ['sin_month', 'cos_month']

    feature_sets = {
        "Full_Set": SPEI_FEATURES + SPI_FEATURES + all_weather_hydr,
        "Decorrelated_Features": decorrelated_features,
        "SPEI_only": SPEI_FEATURES,
        "SPI_only": SPI_FEATURES,
        "SPEI_Weather": SPEI_FEATURES + WEATHER_FEATURES,
        "SPEI_Hydr": SPEI_FEATURES + HYDR_FEATURES,
        "Weather_only": WEATHER_FEATURES,
        "Hydr_only": HYDR_FEATURES,
        "Weather_Hydr": WEATHER_FEATURES + HYDR_FEATURES,
        "Weather_Geo": WEATHER_FEATURES + GEO_FEATURES,
        "Hydr_Geo": HYDR_FEATURES + GEO_FEATURES,
        "Geography_only": GEO_FEATURES,
        "Seasonal": ['sin_month', 'cos_month'] + oh_cols,
    }
    return feature_sets

def train_and_evaluate_feature_set(X_train_full, y_train_full, X_test, y_test, features, name):
    X_train = X_train_full[features]
    X_test_final = X_test[features]

    rf_selector = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_final = RandomForestClassifier(**RF_PARAMS)
    
    model_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampling', SMOTE(random_state=SMOTE_RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)),
        ('feature_selection', SelectFromModel(rf_selector)),
        ('classifier', rf_final)
    ])

    model_pipeline.fit(X_train, y_train_full)
    
    y_pred_train = model_pipeline.predict(X_train)
    y_pred_test  = model_pipeline.predict(X_test_final)
    y_prob_train = model_pipeline.predict_proba(X_train)[:, 1]
    y_prob_test  = model_pipeline.predict_proba(X_test_final)[:, 1]
    
    return {
        'Feature_Set': name,
        'Train_Accuracy': accuracy_score(y_train_full, y_pred_train),
        'Test_Accuracy': accuracy_score(y_test, y_pred_test),
        'Train_F1': f1_score(y_train_full, y_pred_train, zero_division=0),
        'Test_F1': f1_score(y_test, y_pred_test, zero_division=0),
        'Train_ROC_AUC': roc_auc_score(y_train_full, y_prob_train),
        'Test_ROC_AUC': roc_auc_score(y_test, y_prob_test)
    }

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    training_data = pd.read_csv(TRAIN_FILE)
    testing_data  = pd.read_csv(TEST_FILE)
    
    training_data = generate_features(training_data)
    testing_data  = generate_features(testing_data)
    
    y_train = training_data[TARGET_COLUMN]
    y_test  = testing_data[TARGET_COLUMN]
    
    cols_to_drop = [TARGET_COLUMN, 'Provincia', 'Codice', 'Month']
    X_train_full = training_data.drop(columns=[c for c in cols_to_drop if c in training_data.columns])
    X_test_full  = testing_data.drop(columns=[c for c in cols_to_drop if c in testing_data.columns])
    
    all_features = list(set(X_train_full.columns) | set(X_test_full.columns))
    X_train_full = X_train_full.reindex(columns=all_features, fill_value=0)
    X_test_full = X_test_full.reindex(columns=all_features, fill_value=0)
    
    feature_sets = get_base_feature_sets(training_data)
    all_results = []
    print(f"\nTraining and evaluating {len(feature_sets)} feature sets...\n")

    for name, features in feature_sets.items():
        valid_features = [f for f in features if f in X_train_full.columns]
        if valid_features:
            metrics = train_and_evaluate_feature_set(X_train_full, y_train, X_test_full, y_test, valid_features, name)
            all_results.append(metrics)
            
    results_df = pd.DataFrame(all_results)
    print(results_df)

# Generalization 
import numpy as np
import pandas as pd
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

TRAIN_FILE = r"Train- without Nan.csv"
PREDICT_FILE = r"ready.csv"
TARGET_COLUMN = 'Drought'
RANDOM_STATE = 42

RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}

CATEGORICAL_COLUMNS = ['Provincia', 'Codice']
SPEI_FEATURES = ['SPEI01','SPEI03','SPEI06','SPEI09','SPEI12']
SPI_FEATURES  = ['SPI01','SPI03','SPI06','SPI09','SPI12']
HYDR_FEATURES = ['gr','if','rf','ws']
WEATHER_FEATURES = ['ae','pr','td','tp']

def generate_features(df):
    for c in CATEGORICAL_COLUMNS:
        if c in df: df[c] = df[c].astype(str)
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLUMNS if c in df], drop_first=True)
    if 'Month' in df:
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(1).astype(int)
        df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.fillna(0)

def feature_set(df):
    oh = [c for c in df if any(cc in c for cc in CATEGORICAL_COLUMNS)]
    seas = ['sin_month','cos_month'] if 'sin_month' in df else []
    f = list(set(SPEI_FEATURES + SPI_FEATURES + WEATHER_FEATURES + HYDR_FEATURES + oh + seas))
    return [x for x in f if x in df]

def pipeline():
    return ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=2)),
        ('select', SelectFromModel(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))),
        ('clf', RandomForestClassifier(**RF_PARAMS))
    ])

def train(X, y, f):
    m = pipeline()
    m.fit(X[f], y)
    return m

def predict_save(m, X, f, path):
    Xp = X[f]
    probs = m.predict_proba(Xp)[:,1]
    preds = m.predict(Xp)
    res = pd.DataFrame({'Predicted_Drought_Label': preds, 'Drought_Probability': probs}, index=X.index)
    orig = pd.read_csv(path)
    out = os.path.join(os.path.dirname(path), os.path.basename(path).replace('.csv', '_predicted.csv'))
    pd.concat([orig.reset_index(drop=True), res], axis=1).to_csv(out, index=False)
    print(f"Saved: {out}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_df = generate_features(pd.read_csv(TRAIN_FILE))
    y = train_df[TARGET_COLUMN]
    X = train_df.drop(columns=[c for c in [TARGET_COLUMN, 'Provincia', 'Codice', 'Month'] if c in train_df])
    feats = feature_set(train_df)
    pred_df = generate_features(pd.read_csv(PREDICT_FILE))
    Xp = pred_df.drop(columns=[c for c in ['Provincia', 'Codice', 'Month'] if c in pred_df], errors='ignore')
    allcols = list(set(X.columns) | set(Xp.columns))
    X, Xp = X.reindex(columns=allcols, fill_value=0), Xp.reindex(columns=allcols, fill_value=0)
    feats = [f for f in feats if f in X]
    if feats:
        model = train(X, y, feats)
        predict_save(model, Xp, feats, PREDICT_FILE)
    else:
        print("No features found.")

# Mapping
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.spatial import cKDTree
from matplotlib.colors import Normalize
from tqdm import tqdm
from joblib import Parallel, delayed

csv_path = r"Mean Annual Drought Probability.csv"
shapefile_path = r"shape file.shp"

df = pd.read_csv(csv_path)
gdf_shape = gpd.read_file(shapefile_path)

geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

if gdf_points.crs != gdf_shape.crs:
    gdf_points = gdf_points.to_crs(gdf_shape.crs)

minx, miny, maxx, maxy = gdf_shape.total_bounds
num_cols, num_rows = 120, 120
xi = np.linspace(minx, maxx, num_cols)
yi = np.linspace(miny, maxy, num_rows)
xi, yi = np.meshgrid(xi, yi)
grid_points = np.column_stack((xi.ravel(), yi.ravel()))

grid_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in grid_points], crs=gdf_shape.crs)
mask_geom = gdf_shape.geometry.unary_union
mask = grid_gdf.within(mask_geom).values
grid_points_masked = grid_points[mask]

def idw(xy, values, xi, power=2, k=4):
    tree = cKDTree(xy)
    dists, idxs = tree.query(xi, k=k)
    weights = 1 / (dists + 1e-12) ** power
    weights /= weights.sum(axis=1)[:, None]
    zi = np.sum(weights * values[idxs], axis=1)
    return zi

years = sorted(df['Year'].unique())
n_years = len(years)
cols = 4
rows = int(np.ceil(n_years / cols))
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
axs = axs.ravel()

norm = Normalize(vmin=df['Mean Annual Drought Probability'].min(),
                 vmax=df['Mean Annual Drought Probability'].max())

def process_year(year):
    sub = gdf_points[gdf_points['Year'] == year]
    if sub.empty:
        return None
    xy = np.array(list(zip(sub.geometry.x, sub.geometry.y)))
    values = sub['Mean Annual Drought Probability'].values
    zi_masked = idw(xy, values, grid_points_masked)
    zi = np.full(xi.shape, np.nan)
    zi.ravel()[mask] = zi_masked
    return year, zi

results = Parallel(n_jobs=-1)(delayed(process_year)(year) for year in tqdm(years, desc="Processing years"))

plot_idx = 0
for res in results:
    if res is None:
        continue
    year, zi = res
    ax = axs[plot_idx]
    im = ax.imshow(zi, extent=(minx, maxx, miny, maxy), origin='lower', cmap='YlOrBr', norm=norm)
    gdf_shape.boundary.plot(ax=ax, color='black', linewidth=0.5)
    ax.set_title(f"Year: {year}", fontsize=16, fontweight='bold')
    ax.axis('off')
    plot_idx += 1
for i in range(plot_idx, len(axs)):
    axs[i].axis('off')
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.2])
cbar = fig.colorbar(im, cax=cbar_ax, label='Mean Annual Drought Probability')
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.label.set_size(16)
plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.05)
plt.savefig(r"Mean_Annual_Drought_Probability.jpg",
            dpi=600, bbox_inches='tight')
plt.show()

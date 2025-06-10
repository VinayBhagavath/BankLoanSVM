import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv('bankloans.csv')
except FileNotFoundError:
    print("The file 'bankloans.csv' was not found. Please make sure it is in the correct directory.")
    exit()

# Drop rows where the 'default' status is unknown
df.dropna(subset=['default'], inplace=True)

# Define features (X) and target (y)
X = df.drop('default', axis=1)
y = df['default']

# --- 2. Preprocess and Reduce Dimensionality ---

# We will only use numerical features for this visualization
numerical_features = X.select_dtypes(include=np.number).columns
X_numerical = X[numerical_features]

# Step 2a: Scale the numerical features
# This is crucial for both PCA and SVM to work correctly.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

# Step 2b: Apply PCA to reduce dimensions from 8 to 3
# We are creating a new 3D representation of the data.
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print(f"Original data shape: {X_scaled.shape}")
print(f"Data shape after PCA: {X_pca.shape}")


# --- 3. Train a New SVM on the 3D Data ---
# We train a new model specifically for this 3D space to find the
# corresponding separating plane.
svm_3d = SVC(kernel='linear')
svm_3d.fit(X_pca, y)


# --- 4. Prepare for Visualization ---

# Get the coefficients (w) and intercept (b) of the hyperplane
# The equation of the plane is w0*x + w1*y + w2*z + b = 0
weights = svm_3d.coef_[0]
intercept = svm_3d.intercept_[0]

# Create a meshgrid to plot the hyperplane
# We need to create a grid of x and y points, then calculate the corresponding z
# to form the plane.
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

# Create a grid of points
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                     np.linspace(y_min, y_max, 50))

# Calculate the corresponding z value for each point on the grid using the plane equation
# z = (-w0*x - w1*y - b) / w2
zz = (-weights[0] * xx - weights[1] * yy - intercept) / weights[2]


# --- 5. Create the 3D Plot using Plotly ---

# Create a figure
fig = go.Figure()

# Add the hyperplane as a surface
fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='Blues',
              opacity=0.6, showscale=False, name='Hyperplane'))

# Add the data points as a scatter plot
# We separate them by class (default or not) to color them differently.
fig.add_trace(go.Scatter3d(
    x=X_pca[y == 0, 0],
    y=X_pca[y == 0, 1],
    z=X_pca[y == 0, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='green',  # Non-defaulters
        opacity=0.8
    ),
    name='Non-Default'
))

fig.add_trace(go.Scatter3d(
    x=X_pca[y == 1, 0],
    y=X_pca[y == 1, 1],
    z=X_pca[y == 1, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='red',  # Defaulters
        opacity=0.8
    ),
    name='Default'
))

# Update the layout for a clean look
fig.update_layout(
    title_text='3D Visualization of SVM Hyperplane and Data Points',
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# Show the interactive plot
fig.show()

# flask_app.py
from flask import Flask, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import json
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)
CORS(app)


@app.route('/plot-data')
def plot_data():
    df = pd.read_csv('bankloans.csv')
    df.dropna(subset=['default'], inplace=True)

    X = df.drop('default', axis=1)
    y = df['default']
    X_numerical = X.select_dtypes(include=np.number)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numerical)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    svm_3d = SVC(kernel='linear')
    svm_3d.fit(X_pca, y)

    weights = svm_3d.coef_[0]
    intercept = svm_3d.intercept_[0]

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    zz = (-weights[0] * xx - weights[1] * yy - intercept) / weights[2]

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz,
                  colorscale='Blues', opacity=0.6, showscale=False))
    fig.add_trace(go.Scatter3d(
        x=X_pca[y == 0, 0], y=X_pca[y == 0, 1], z=X_pca[y == 0, 2],
        mode='markers', marker=dict(size=5, color='green', opacity=0.8),
        name='Non-Default'))
    fig.add_trace(go.Scatter3d(
        x=X_pca[y == 1, 0], y=X_pca[y == 1, 1], z=X_pca[y == 1, 2],
        mode='markers', marker=dict(size=5, color='red', opacity=0.8),
        name='Default'))

    fig.update_layout(
        title='3D Visualization of SVM Hyperplane and Data Points',
        scene=dict(zaxis=dict(
            range=[-10, 20]), xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig_json_str = json.dumps(fig, cls=PlotlyJSONEncoder)
    return Response(fig_json_str, mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)

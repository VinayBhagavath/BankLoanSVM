import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

function App() {
  const [plotData, setPlotData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Replace with the correct backend address if different
    fetch('http://localhost:5000/plot-data')
      .then((res) => res.json())
      .then((data) => {
        setPlotData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to fetch plot data:", err);
        setLoading(false);
      });
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h2>3D SVM Hyperplane Visualization</h2>
      {loading ? (
        <p>Loading plot...</p>
      ) : plotData ? (
        <Plot
          data={plotData.data}
          layout={plotData.layout}
          config={{ responsive: true }}
          style={{ width: '100%', height: '80vh' }}
        />
      ) : (
        <p>Failed to load plot.</p>
      )}
    </div>
  );
}

export default App;

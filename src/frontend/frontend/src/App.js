// === App.js ===
import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import { io } from 'socket.io-client';
import CanvasHeatmap from './CanvasHeatmap';
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  LogarithmicScale,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, LogarithmicScale, Tooltip, Legend, Filler);

const socket = io("http://localhost:8000");

function App() {
  const [dataPoints, setDataPoints] = useState([]);
  const [labels, setLabels] = useState([]);
  const [pred, setPred] = useState([]);

  useEffect(() => {
    socket.on("new_loss", (data) => {
      console.log("Nuevo loss recibido:", data.loss);
      setDataPoints(prev => [...prev, data.loss]);
      setLabels(prev => [...prev, prev.length]);
    });

    socket.on("new_pred", (data) => {
      console.log("Nueva predicción recibida");
      setPred(data.pred);
    });

    return () => {
      socket.off("new_loss");
      socket.off("new_pred");
    };
  }, []);

  const chartData = {
    labels: labels,
    datasets: [
      {
        label: 'Loss',
        data: dataPoints,
        borderColor: 'blue',
        backgroundColor: 'blue',
        pointRadius: 3,
        pointBorderWidth: 2,
        tension: 0.1,
        fill: false
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        type: 'logarithmic',
        beginAtZero: false,
        ticks: {
          callback: function (value) {
            return Number(value).toExponential(1);
          },
          color: 'black'
        },
        title: {
          display: true,
          text: 'Loss (log)',
          color: 'black'
        }
      },
      x: {
        ticks: {
          color: 'black'
        },
        title: {
          display: true,
          text: 'Epoch',
          color: 'black'
        }
      }
    },
    plugins: {
      legend: {
        display: false
      }
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', padding: '10px', boxSizing: 'border-box', display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr', gap: '10px' }}>
      {/* Cuadro superior izquierdo - Gráfico de Loss */}
      <div style={{ gridColumn: '1', gridRow: '1', backgroundColor: '#fff', border: '1px solid #ccc', padding: '10px', overflow: 'hidden' }}>
        <h4 style={{ margin: '0 0 10px 0' }}>Loss en tiempo real</h4>
        <div style={{ width: '100%', height: '100%' }}>
          <Line data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* Cuadro inferior izquierdo - Predicción */}
      {pred.length > 0 && (
        <div style={{ gridColumn: '1', gridRow: '2', backgroundColor: '#fff', border: '1px solid #ccc', padding: '10px', overflow: 'hidden' }}>
          <h4 style={{ marginBottom: '10px' }}>Predicción en tiempo real</h4>
          <div style={{ width: '100%', height: '100%' }}>
            <CanvasHeatmap data={pred} />
          </div>
        </div>
      )}

      {/* Cuadro superior derecho - Historial de Loss */}
      <div style={{ gridColumn: '2', gridRow: '1', backgroundColor: '#fff', border: '1px solid #ccc', padding: '10px', overflowY: 'auto' }}>
        <h4 style={{ marginBottom: '10px' }}>Historial de Loss</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', borderBottom: '1px solid #ccc' }}>Epoch</th>
              <th style={{ textAlign: 'left', borderBottom: '1px solid #ccc' }}>Loss</th>
            </tr>
          </thead>
          <tbody>
            {dataPoints.map((value, index) => (
              <tr key={index}>
                <td style={{ padding: '4px 8px' }}>{index}</td>
                <td style={{ padding: '4px 0' }}>{value.toFixed(16)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Cuadro inferior derecho - Espacio reservado para futuros controles */}
      <div style={{ gridColumn: '2', gridRow: '2', backgroundColor: '#f9f9f9', border: '1px solid #ccc', padding: '10px' }}>
        <h4>Controles</h4>
        <p>Espacio reservado para botones o configuraciones futuras.</p>
      </div>
    </div>
  );
}

export default App;
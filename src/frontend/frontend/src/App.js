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
  const [opt, setOpt] = useState("rmsprop");
  const [beta, setBeta] = useState(0.9);
  const [numEpochs, setNumEpochs] = useState(10);
  const [stepSize, setStepSize] = useState(0.001);
  const [alpha, setAlpha] = useState(0.02);
  const [batchSize, setBatchSize] = useState(32);
  const [r, setR] = useState(0.0);
  const [s, setS] = useState(0.0);

  useEffect(() => {
    socket.on("new_loss", (data) => {
      setDataPoints(prev => [...prev, data.loss]);
      setLabels(prev => [...prev, prev.length]);
    });

    socket.on("new_pred", (data) => {
      setPred(data.pred);
    });

    return () => {
      socket.off("new_loss");
      socket.off("new_pred");
    };
  }, []);

  const iniciarEntrenamiento = () => {
    const payload = {
      opt,
      num_epochs: Number(numEpochs),
      step_size: Number(stepSize),
      alpha: Number(alpha),
      batch_size: Number(batchSize),
      beta: Number(beta),
      r: Number(r),
      s: Number(s)
    };

    fetch("http://localhost:8000/entrenar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    setDataPoints([]);
    setLabels([]);
  };

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
        type: 'linear',
        beginAtZero: false,
        suggestedMin: Math.min(...dataPoints || [0.001]) * 0.9,
        suggestedMax: Math.max(...dataPoints || [0.1]) * 1.1,
        ticks: {
          color: 'black',
          callback: function (value) {
            return Number(value).toExponential(1);
          }
        },
        title: {
          display: true,
          text: 'Loss',
          color: 'black'
        }
      },
      x: {
        ticks: {
          color: 'black',
          callback: function (val, index, ticks) {
            if (index >= ticks.length - 5) return val;
            return index % 3 === 0 ? val : '';
          }
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
  <div style={{
    width: '100vw',
    height: '100vh',
    padding: '10px',
    boxSizing: 'border-box',
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gridTemplateRows: '1fr 1fr',
    gap: '10px'
  }}>
    {/* Gráfico de pérdida */}
    <div style={{
      gridColumn: '1',
      gridRow: '1',
      backgroundColor: '#fff',
      border: '1px solid #ccc',
      padding: '10px',
      overflow: 'hidden'
    }}>
      <h4>Loss en tiempo real</h4>
      <div style={{ height: '85%', width: '100%' }}>
        <Line data={chartData} options={chartOptions} />
      </div>
    </div>

    {/* Mapa de calor */}
    {Array.isArray(pred) && pred.length > 0 && Array.isArray(pred[0]) && (
      <div style={{
        gridColumn: '1',
        gridRow: '2',
        backgroundColor: '#fff',
        border: '1px solid #ccc',
        padding: '10px',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <h4>Predicción en tiempo real</h4>
        <div style={{ flexGrow: 1, height: '100%', width: '100%' }}>
          <CanvasHeatmap data={pred} />
        </div>
      </div>
    )}

    {/* Tabla historial */}
    <div style={{
      gridColumn: '2',
      gridRow: '1',
      backgroundColor: '#fff',
      border: '1px solid #ccc',
      padding: '10px',
      overflowY: 'auto'
    }}>
      <h4>Historial de Loss</h4>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left' }}>Epoch</th>
            <th style={{ textAlign: 'left' }}>Loss</th>
          </tr>
        </thead>
        <tbody>
          {dataPoints.map((value, index) => (
            <tr key={index}>
              <td>{index}</td>
              <td>{value.toFixed(16)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>

    {/* Controles */}
    <div style={{
      gridColumn: '2',
      gridRow: '2',
      backgroundColor: '#f9f9f9',
      border: '1px solid #ccc',
      padding: '10px',
      overflowY: 'auto'
    }}>
      <h4>Controles</h4>

      <div style={{ marginBottom: '10px' }}>
        <label><input type="radio" name="opt" value="rmsprop" checked={opt === "rmsprop"} onChange={e => setOpt(e.target.value)} /> RMSProp</label><br />
        <label><input type="radio" name="opt" value="sgd" checked={opt === "sgd"} onChange={e => setOpt(e.target.value)} /> SGD</label><br />
        <label><input type="radio" name="opt" value="momentum" checked={opt === "momentum"} onChange={e => setOpt(e.target.value)} /> SGD + Momentum</label><br />
        <label><input type="radio" name="opt" value="adam" checked={opt === "adam"} onChange={e => setOpt(e.target.value)} /> Adam</label>
      </div>

{/* Sliders de parámetros */}
<div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
  <label>Epochs: {numEpochs}
    <input type="range" min="1" max="1000" step="1" value={numEpochs} onChange={e => setNumEpochs(e.target.value)} />
  </label>

  <label>Step Size: {stepSize}
    <input type="range" min="0.0001" max="0.1" step="0.0001" value={stepSize} onChange={e => setStepSize(e.target.value)} />
  </label>

  <label>Alpha: {alpha}
    <input type="range" min="0.001" max="0.1" step="0.001" value={alpha} onChange={e => setAlpha(e.target.value)} />
  </label>

  <label>Batch Size: {batchSize}
    <input type="range" min="1" max="128" step="1" value={batchSize} onChange={e => setBatchSize(e.target.value)} />
  </label>
  {opt === "momentum" && (
  <label>Beta (Momentum): {beta}

  <input type="range" min="0.0001" max="5" step="0.0001" value={beta} onChange={e => setBeta(e.target.value)} />
  </label>
   )}

  {opt === "adam" && (
    <>
      <label>r (Adam): {r}
        <input type="range" min="0.0" max="1.0" step="0.01" value={r} onChange={e => setR(e.target.value)} />
      </label>
      <label>s (Adam): {s}
        <input type="range" min="0.0" max="1.0" step="0.01" value={s} onChange={e => setS(e.target.value)} />
      </label>
    </>
  )}
</div>


      <button onClick={iniciarEntrenamiento} style={{
        marginTop: "10px",
        padding: "10px 20px",
        backgroundColor: "#0074D9",
        color: "white",
        border: "none",
        borderRadius: "4px"
      }}>
        Iniciar entrenamiento
      </button>
      <button
      onClick={() => {fetch("http://localhost:8000/reset", { method: "POST" }); }}
      style={{
        marginTop: "10px",
        padding: "10px 20px",
        backgroundColor: "#FF4136",
        color: "white",
        border: "none",
        borderRadius: "4px" }}>
      Stop
    </button>
    </div>
  </div>
);

}

export default App;

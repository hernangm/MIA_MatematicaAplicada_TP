// CanvasHeatmap.js
import React, { useRef, useEffect } from 'react';

function CanvasHeatmap({ data }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const width = data[0].length;
    const height = data.length;
    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);

    // Normalizar valores para escala de color
    const flat = data.flat();
    const min = Math.min(...flat);
    const max = Math.max(...flat);
    const normalize = val => (val - min) / (max - min + 1e-8);

    const jetColorMap = val => {
      const fourValue = 4 * val;
      const r = Math.min(fourValue - 1.5, -fourValue + 4.5);
      const g = Math.min(fourValue - 0.5, -fourValue + 3.5);
      const b = Math.min(fourValue + 0.5, -fourValue + 2.5);
      return [
        Math.max(0, Math.min(1, r)) * 255,
        Math.max(0, Math.min(1, g)) * 255,
        Math.max(0, Math.min(1, b)) * 255
      ];
    };

    let i = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const norm = normalize(data[y][x]);
        const [r, g, b] = jetColorMap(norm);
        imageData.data[i++] = r;
        imageData.data[i++] = g;
        imageData.data[i++] = b;
        imageData.data[i++] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data]);

  return (
    <canvas
  ref={canvasRef}
  style={{
    width: '100%',
    height: 'auto',
    aspectRatio: '1 / 1',
    maxHeight: '100%',
    border: '1px solid #ccc'
  }}
/>
  );
}

export default CanvasHeatmap;

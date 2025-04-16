import React, { useState } from 'react';
import { analyzeImage } from '../services/api';

const ImageAnalyzer = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const response = await analyzeImage(file);
      setResult(response.result);
    } catch (err) {
      setError('Failed to analyze image. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="image-analyzer">
      <input 
        type="file" 
        accept="image/*"
        onChange={handleImageUpload}
        disabled={loading}
      />
      {loading && <div>Analyzing image...</div>}
      {error && <div className="error">{error}</div>}
      {result && (
        <div className="results">
          <h2>Analysis Results</h2>
          <p>Classification: {result.classification}</p>
          <p>Confidence: {result.confidence_score.toFixed(2)}%</p>
          <p>{result.base_explanation}</p>
        </div>
      )}
    </div>
  );
};

export default ImageAnalyzer;
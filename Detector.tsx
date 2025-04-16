import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { analyzeImage, AnalysisResult } from '../services/api.ts';
import './Detector.css';

const Detector: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [collapsedSections, setCollapsedSections] = useState<{[key: string]: boolean}>({
        'detailed': true,
        'technical': true
    });

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setPreview(URL.createObjectURL(file));
            setResult(null);
            setError(null);
        }
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;
        
        setLoading(true);
        setError(null);
        setResult(null);
        
        try {
            const response = await analyzeImage(selectedFile);
            if (response.success && response.result) {
                setResult(response.result);
            } else {
                throw new Error(response.error || 'Analysis failed');
            }
        } catch (error) {
            setError(error instanceof Error ? error.message : 'An unexpected error occurred');
            console.error('Analysis error:', error);
        } finally {
            setLoading(false);
        }
    };

    const toggleSection = (section: string) => {
        setCollapsedSections(prev => ({
            ...prev,
            [section]: !prev[section]
        }));
    };

    const renderDetailedAnalysis = (text?: string) => {
        if (!text) return null;
        
        // Clean up the text
        const cleanText = text.trim();
        
        // Split into paragraphs
        const paragraphs = cleanText.split('\n\n').filter(p => p.trim());
        
        return (
            <div className="detailed-analysis">
                {paragraphs.map((paragraph, index) => {
                    // Handle headings with **
                    if (paragraph.match(/^\*\*.*\*\*$/)) {
                        return (
                            <h4 key={index} className="analysis-heading">
                                {paragraph.replace(/^\*\*|\*\*$/g, '')}
                            </h4>
                        );
                    }
                    
                    // Handle paragraphs with bold sections
                    const parts = paragraph.split(/(\*\*.*?\*\*)/g);
                    
                    return (
                        <p key={index} className="analysis-paragraph">
                            {parts.map((part, partIndex) => {
                                if (part.startsWith('**') && part.endsWith('**')) {
                                    return (
                                        <strong key={partIndex}>
                                            {part.replace(/^\*\*|\*\*$/g, '')}
                                        </strong>
                                    );
                                }
                                return <span key={partIndex}>{part}</span>;
                            })}
                        </p>
                    );
                })}
            </div>
        );
    };

    const renderMetricCard = (title: string, value: number | undefined | null, suffix: string = '') => {
        if (typeof value !== 'number') return null;
        
        const formattedValue = Number.isFinite(value) ? value.toFixed(2) : 'N/A';
                
        return (
            <div className="metric-item">
                <label>{title}</label>
                <div className="progress-bar">
                    <div 
                        className="progress-fill"
                        style={{ 
                            width: `${Number.isFinite(value) ? Math.min(100, Math.max(0, value)) : 0}%` 
                        }}
                    />
                </div>
                <span>{formattedValue}{suffix}</span>
            </div>
        );
    };

    const formatNumber = (value: number | undefined | null): string => {
        if (typeof value !== 'number' || !Number.isFinite(value)) return 'N/A';
        return value.toFixed(2);
    };

    const renderAnalysisResult = () => {
        if (!result) return null;

        return (
            <motion.div 
                className="result-section"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5 }}
            >
                <h3>Analysis Result</h3>
                <div className={`result-box ${result.classification === 'FAKE' ? 'fake' : 'authentic'}`}>
                    <div className="result-header">
                        <h4>{result.classification}</h4>
                        {result.confidence && (
                            <div className="confidence-badge">
                                {result.confidence} Confidence ({formatNumber(result.confidence_score)}%)
                            </div>
                        )}
                    </div>

                    {result.feature_analysis?.metrics && (
                        <div className="metrics-grid">
                            <div className="metric-card">
                                <h6>Primary Metrics</h6>
                                {renderMetricCard('Authenticity Score', result.feature_analysis.metrics.authenticity_score, '%')}
                                {renderMetricCard('Consistency Score', result.feature_analysis.metrics.consistency_score, '%')}
                                {renderMetricCard('Complexity Score', result.feature_analysis.metrics.complexity_score, '%')}
                            </div>
                        </div>
                    )}

                    {result.feature_analysis?.activation_patterns && (
                        <div className="metrics-grid">
                            <div className="metric-card">
                                <h6>Technical Metrics</h6>
                                <ul className="metrics-list">
                                    <li>Mean: {formatNumber(result.feature_analysis.activation_patterns.mean)}</li>
                                    <li>Max: {formatNumber(result.feature_analysis.activation_patterns.max)}</li>
                                    <li>Standard Deviation: {formatNumber(result.feature_analysis.activation_patterns.std)}</li>
                                    <li>Entropy: {formatNumber(result.feature_analysis.activation_patterns.entropy)}</li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {result.visualization_data && (
                        <div className="analysis-section">
                            <h5>Visualization Metrics</h5>
                            <div className="metrics-grid">
                                {result.visualization_data.distribution && (
                                    <div className="metric-card">
                                        <h6>Distribution</h6>
                                        <ul className="metrics-list">
                                            <li>Low: {formatNumber(result.visualization_data.distribution.low)}</li>
                                            <li>Medium: {formatNumber(result.visualization_data.distribution.medium)}</li>
                                            <li>High: {formatNumber(result.visualization_data.distribution.high)}</li>
                                        </ul>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {result.detailed_analysis && (
                        <div className="analysis-section">
                            <h5>Detailed Analysis</h5>
                            <button 
                                className="collapse-button"
                                onClick={() => toggleSection('detailed')}
                            >
                                <i className={`fas fa-chevron-${collapsedSections['detailed'] ? 'down' : 'up'}`} />
                                {collapsedSections['detailed'] ? 'Show More' : 'Show Less'}
                            </button>
                            <div className={collapsedSections['detailed'] ? 'collapsed' : ''}>
                                {renderDetailedAnalysis(result.detailed_analysis)}
                            </div>
                        </div>
                    )}
                </div>
            </motion.div>
        );
    };

    return (
        <motion.div 
            className="detector-container"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
        >
            <div className="upload-section">
                <motion.h2 
                    initial={{ y: -20 }}
                    animate={{ y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    AI Image Forgery Detection
                </motion.h2>
                
                <motion.div 
                    className="upload-area"
                    whileHover={{ scale: 1.02 }}
                    onClick={() => document.getElementById('file-input')?.click()}
                >
                    {preview ? (
                        <motion.img 
                            src={preview} 
                            alt="Preview" 
                            className="image-preview"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.5 }}
                        />
                    ) : (
                        <div className="upload-placeholder">
                            <i className="fas fa-cloud-upload-alt"></i>
                            <p>Click or drag to upload image</p>
                        </div>
                    )}
                    <input
                        type="file"
                        id="file-input"
                        accept="image/*"
                        onChange={handleFileSelect}
                        style={{ display: 'none' }}
                    />
                </motion.div>

                {selectedFile && (
                    <motion.button
                        className="analyze-button"
                        onClick={handleAnalyze}
                        disabled={loading}
                        whileHover={{ scale: loading ? 1 : 1.05 }}
                    >
                        {loading ? (
                            <span className="loading-spinner">
                                <i className="fas fa-spinner fa-spin" />
                                Analyzing...
                            </span>
                        ) : (
                            'Analyze Image'
                        )}
                    </motion.button>
                )}

                {error && (
                    <motion.div 
                        className="error-message"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <i className="fas fa-exclamation-circle" /> {error}
                    </motion.div>
                )}
            </div>

            <AnimatePresence mode="wait">
                {result && !loading && renderAnalysisResult()}
            </AnimatePresence>
        </motion.div>
    );
};

export default Detector;
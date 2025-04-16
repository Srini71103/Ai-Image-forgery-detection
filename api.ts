const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface AnalysisMetrics {
    mean: number;
    max: number;
    std: number;
    entropy: number;
    distribution: {
        low: number;
        medium: number;
        high: number;
    };
}

interface VisualizationData {
    scores: {
        authenticity: number;
        consistency: number;
        complexity: number;
    };
    thresholds: {
        activation: number;
        variation: number;
        entropy: number;
    };
    distribution: {
        low: number;
        medium: number;
        high: number;
    };
}

export interface AnalysisResult {
    classification: string;
    confidence: string;
    confidence_score: number;
    feature_analysis: {
        activation_patterns: AnalysisMetrics;
        metrics: {
            authenticity_score: number;
            consistency_score: number;
            complexity_score: number;
        };
    };
    vgg_analysis: {
        summary: string;
        metrics: {
            activation_level: number;
            variation: number;
            entropy: number;
        };
    };
    detailed_analysis: string;
    visualization_data: VisualizationData;
}

export interface ApiResponse {
    success: boolean;
    result: AnalysisResult | null;
    error?: string;
}

export const analyzeImage = async (file: File): Promise<ApiResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/analyze-image/`, {
            method: 'POST',
            body: formData,
            mode: 'cors',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Analysis failed');
        }

        return {
            success: true,
            result: data.result
        };
    } catch (error) {
        console.error('Analysis failed:', error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'An unexpected error occurred',
            result: null
        };
    }
}
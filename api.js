const API_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-backend-url.vercel.app'
  : 'http://localhost:8000';

export const analyzeImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_URL}/analyze-image/`, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
};
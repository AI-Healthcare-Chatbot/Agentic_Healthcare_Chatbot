import axios from 'axios';

// Set the base URL for API requests - empty for same-domain deployment
const BASE_URL = process.env.REACT_APP_API_URL || '';

// Create an axios instance
const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API for RESTful endpoints
export const chatAPI = {
  // Send a message to the chatbot
  sendMessage: async (message) => {
    try {
      const response = await api.post('/api/chat', { message });
      return response.data;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  },

  // Get conversation history
  getHistory: async () => {
    try {
      const response = await api.get('/api/history');
      return response.data.history;
    } catch (error) {
      console.error('Error getting history:', error);
      throw error;
    }
  },

  // Reset conversation
  resetConversation: async () => {
    try {
      const response = await api.post('/api/reset');
      return response.data;
    } catch (error) {
      console.error('Error resetting conversation:', error);
      throw error;
    }
  },
  
  // Get the full URL for a plot
  getPlotUrl: (plotPath) => {
    if (!plotPath) return null;
    
    // If it's already a full URL, return it
    if (plotPath.startsWith('http')) return plotPath;
    
    // Combine with base URL if needed
    return `${BASE_URL}${plotPath.startsWith('/') ? '' : '/'}${plotPath}`;
  }
};

export default { chatAPI };
import axios from 'axios';

// Set the base URL for API requests
const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
  
  // Check if a plot file exists
  checkPlotExists: async (filename) => {
    try {
      const response = await api.get(`/check-plot/${filename}`);
      return response.data.exists;
    } catch (error) {
      console.error('Error checking plot:', error);
      return false;
    }
  },
  
  // Get the full URL for a plot
  getPlotUrl: (plotPath) => {
    if (!plotPath) return null;
    
    // If it's already a full URL, return it
    if (plotPath.startsWith('http')) return plotPath;
    
    // If it's a relative path, combine with base URL
    // Handle both Unix and Windows style paths
    const normalizedPath = plotPath.replace(/\\/g, '/');
    const filename = normalizedPath.split('/').pop();
    return `${BASE_URL}/plots/${filename}`;
  }
};

// WebSocket connection management
let socket = null;
let messageCallbacks = [];

export const chatSocket = {
  // Connect to WebSocket
  connect: (clientId) => {
    if (socket) {
      socket.close();
    }

    const wsUrl = BASE_URL.replace(/^http/, 'ws');
    socket = new WebSocket(`${wsUrl}/ws/${clientId}`);
    
    console.log(`Connecting to WebSocket at ${wsUrl}/ws/${clientId}`);

    socket.onopen = () => {
      console.log('WebSocket connected');
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket message received:', data);
        messageCallbacks.forEach(callback => callback(data));
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    socket.onclose = () => {
      console.log('WebSocket disconnected');
    };

    return {
      // Send a message through WebSocket
      sendMessage: (message) => {
        if (socket.readyState === WebSocket.OPEN) {
          console.log('Sending message via WebSocket:', message);
          socket.send(JSON.stringify({ message }));
        } else {
          console.error('WebSocket not connected, readyState:', socket.readyState);
        }
      },

      // Reset conversation through WebSocket
      resetConversation: () => {
        if (socket.readyState === WebSocket.OPEN) {
          console.log('Resetting conversation via WebSocket');
          socket.send(JSON.stringify({ reset: true }));
        } else {
          console.error('WebSocket not connected for reset');
        }
      },

      // Add a message callback
      addMessageCallback: (callback) => {
        messageCallbacks.push(callback);
      },

      // Remove a message callback
      removeMessageCallback: (callback) => {
        messageCallbacks = messageCallbacks.filter(cb => cb !== callback);
      },

      // Check connection status
      isConnected: () => {
        return socket && socket.readyState === WebSocket.OPEN;
      },

      // Disconnect WebSocket
      disconnect: () => {
        if (socket) {
          socket.close();
          socket = null;
        }
      }
    };
  }
};

export default { chatAPI, chatSocket };
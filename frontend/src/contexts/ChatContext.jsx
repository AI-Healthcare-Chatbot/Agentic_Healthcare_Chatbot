import React, { createContext, useState, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { chatAPI, chatSocket } from '../services/api';

// Create the chat context
export const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
  // State for messages
  const [messages, setMessages] = useState([]);
  // State for loading indicator
  const [isLoading, setIsLoading] = useState(false);
  // Create a client ID for WebSocket connection
  const [clientId] = useState(uuidv4());
  // WebSocket connection object
  const [wsClient, setWsClient] = useState(null);
  // Connection status
  const [isConnected, setIsConnected] = useState(false);
  // Error state
  const [error, setError] = useState(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const client = chatSocket.connect(clientId);
    setWsClient(client);

    // Add message handler
    const handleMessage = (data) => {
      console.log("Received message:", data);
      
      if (data.type === 'thinking') {
        setIsLoading(true);
      } else if (data.type === 'response') {
        setIsLoading(false);
        setMessages(prevMessages => [
          ...prevMessages, 
          { id: uuidv4(), role: 'assistant', content: data.message }
        ]);
      } else if (data.type === 'plot') {
        console.log("Received plot:", data.url);
        // Add the plot to the last assistant message
        setMessages(prevMessages => {
          const lastMessageIndex = prevMessages.length - 1;
          if (lastMessageIndex >= 0 && prevMessages[lastMessageIndex].role === 'assistant') {
            const updatedMessages = [...prevMessages];
            updatedMessages[lastMessageIndex] = {
              ...updatedMessages[lastMessageIndex],
              plot: data.url
            };
            return updatedMessages;
          }
          return prevMessages;
        });
      } else if (data.type === 'error') {
        setIsLoading(false);
        setError(data.message);
      }
    };

    client.addMessageCallback(handleMessage);
    setIsConnected(true);

    // Cleanup function
    return () => {
      client.removeMessageCallback(handleMessage);
      client.disconnect();
      setIsConnected(false);
    };
  }, [clientId]);

  // Reset chat on page refresh (this runs only once on component mount)
  useEffect(() => {
    // Clear chat messages on page load/refresh
    resetConversation();
    
    // Optional: Add event listener for page refresh/reload
    const handleBeforeUnload = () => {
      // This won't affect the next session, but it's good practice for cleanup
      localStorage.removeItem('chatHistory');
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  // Function to check for plot in message and update message
  const checkForPlotInMessage = async (message) => {
    try {
      // Look for both Windows and Unix style paths
      const plotRegex = /plots[\\\/]([a-zA-Z0-9_-]+\.png)/;
      const match = plotRegex.exec(message.content);
      
      if (match) {
        const plotPath = match[1];
        // Normalize path separators
        const normalizedPath = plotPath.replace(/\\/g, '/');
        const plotFilename = normalizedPath.split('/').pop();
        
        // Check if the plot file exists on the server
        const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/check-plot/${plotFilename}`);
        const data = await response.json();
        
        if (data.exists) {
          // Update the message with the plot URL
          const plotUrl = `/plots/${plotFilename}`;
          return {
            ...message,
            content: message.content.replace(match[0], ''),
            plot: plotUrl
          };
        }
      }
      
      return message;
    } catch (error) {
      console.error('Error checking for plot:', error);
      return message;
    }
  };

  // Send a message
  const sendMessage = useCallback(async (message) => {
    if (!message.trim()) return;

    // Optimistically add user message to the UI
    const userMessage = { id: uuidv4(), role: 'user', content: message };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      if (wsClient && isConnected) {
        // Use WebSocket for real-time communication
        wsClient.sendMessage(message);
      } else {
        // Fallback to REST API
        const response = await chatAPI.sendMessage(message);
        setIsLoading(false);
        
        // Add assistant response
        let assistantMessage = { 
          id: uuidv4(), 
          role: 'assistant', 
          content: response.response 
        };
        
        // Add plot if available
        if (response.plot) {
          console.log("Plot received from API:", response.plot);
          assistantMessage.plot = response.plot.url;
        }
        
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      setIsLoading(false);
      setError('Error sending message. Please try again.');
      console.error('Error sending message:', error);
    }
  }, [wsClient, isConnected]);

  // Reset conversation
  const resetConversation = useCallback(async () => {
    try {
      setMessages([]);
      setError(null);
      
      if (wsClient && isConnected) {
        wsClient.resetConversation();
      } else {
        await chatAPI.resetConversation();
      }
    } catch (error) {
      setError('Error resetting conversation. Please try again.');
      console.error('Error resetting conversation:', error);
    }
  }, [wsClient, isConnected]);

  // Context value
  const value = {
    messages,
    isLoading,
    error,
    sendMessage,
    resetConversation
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};
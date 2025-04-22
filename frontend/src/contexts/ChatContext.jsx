import React, { createContext, useState, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { chatAPI } from '../services/api';

// Create the chat context
export const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
  // State for messages
  const [messages, setMessages] = useState([]);
  // State for loading indicator
  const [isLoading, setIsLoading] = useState(false);
  // Error state
  const [error, setError] = useState(null);

  // Reset conversation on page load/refresh
  useEffect(() => {
    resetConversation();
  }, []);

  // Send a message
  const sendMessage = useCallback(async (message) => {
    if (!message.trim()) return;

    // Add user message to UI
    const userMessage = { 
      id: Date.now().toString(), 
      role: 'user', 
      content: message 
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Send to API
      const response = await chatAPI.sendMessage(message);
      setIsLoading(false);
      
      // Add assistant response
      const assistantMessage = { 
        id: (Date.now() + 1).toString(), 
        role: 'assistant', 
        content: response.response 
      };
      
      // Add plot if available
      if (response.plot) {
        assistantMessage.plot = response.plot.url;
      }
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      setIsLoading(false);
      setError('Error sending message. Please try again.');
      console.error('Error sending message:', error);
    }
  }, []);

  // Reset conversation
  const resetConversation = useCallback(async () => {
    try {
      setMessages([]);
      setError(null);
      await chatAPI.resetConversation();
    } catch (error) {
      setError('Error resetting conversation.');
      console.error('Error resetting conversation:', error);
    }
  }, []);

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
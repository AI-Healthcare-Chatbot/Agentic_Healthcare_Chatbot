/**
 * Utility functions for the healthcare chatbot frontend
 */

/**
 * Format a date to a human-readable string
 * @param {Date} date - The date to format
 * @returns {string} - Formatted date string
 */
export const formatDate = (date) => {
  if (!date) return '';
  
  const options = { 
    year: 'numeric', 
    month: 'short', 
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  };
  
  return new Date(date).toLocaleDateString(undefined, options);
};

/**
 * Generate a unique ID for chat messages
 * @returns {string} - Unique ID
 */
export const generateMessageId = () => {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Extract plot URL from bot response if it exists
 * @param {string} response - Bot response text
 * @returns {string|null} - URL of the plot or null if not found
 */
export const extractPlotUrl = (response) => {
  if (!response) return null;
  
  // Check for plot path in the response
  const plotMatch = response.match(/Plot saved and opened as (.+?)(?:\s|$)/);
  if (plotMatch && plotMatch[1]) {
    return plotMatch[1];
  }
  
  return null;
};

/**
 * Sanitize user input to prevent harmful inputs
 * @param {string} input - User input to sanitize
 * @returns {string} - Sanitized input
 */
export const sanitizeInput = (input) => {
  if (!input) return '';
  
  // Basic sanitization - replace HTML tags
  return input
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
};

/**
 * Check if a string contains healthcare-related terms for highlighting
 * @param {string} text - Text to check
 * @returns {boolean} - True if text contains healthcare terms
 */
export const containsHealthcareTerms = (text) => {
  if (!text) return false;
  
  const healthcareTerms = [
    'hospital', 'patient', 'doctor', 'nurse', 'medical', 'clinical',
    'diagnosis', 'treatment', 'therapy', 'procedure', 'surgery',
    'medication', 'drug', 'prescription', 'disease', 'condition',
    'symptoms', 'cure', 'health', 'care', 'insurance', 'medicare',
    'medicaid', 'pharmacy', 'physician', 'clinic', 'admission',
    'discharge', 'icu', 'emergency', 'ambulance', 'vaccine',
    'outbreak', 'epidemic', 'pandemic', 'virus', 'bacteria',
    'infection', 'chronic', 'acute', 'terminal', 'recovery'
  ];
  
  return healthcareTerms.some(term => 
    new RegExp(`\\b${term}\\b`, 'i').test(text)
  );
};

/**
 * Parse and format error messages from the API
 * @param {Error} error - Error object from API call
 * @returns {string} - Formatted error message
 */
export const parseErrorMessage = (error) => {
  if (!error) return 'An unknown error occurred';
  
  if (error.response && error.response.data && error.response.data.detail) {
    return `Error: ${error.response.data.detail}`;
  }
  
  if (error.message) {
    return `Error: ${error.message}`;
  }
  
  return 'An unexpected error occurred. Please try again.';
};

/**
 * Detect if the current browser supports speech recognition
 * @returns {boolean} - True if speech recognition is supported
 */
export const isSpeechRecognitionSupported = () => {
  return 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
};

/**
 * Check if a message contains a request for a plot
 * @param {string} message - Message to check
 * @returns {boolean} - True if message contains plot request
 */
export const isPlotRequest = (message) => {
  if (!message) return false;
  
  const plotKeywords = [
    'plot', 'chart', 'graph', 'visualize', 'visualization',
    'show me', 'display', 'diagram'
  ];
  
  return plotKeywords.some(keyword => 
    message.toLowerCase().includes(keyword)
  );
};

/**
 * Format large numbers with commas
 * @param {number} num - Number to format
 * @returns {string} - Formatted number string
 */
export const formatNumber = (num) => {
  if (num === null || num === undefined) return '';
  
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
};

/**
 * Format currency values with dollar sign and commas
 * @param {number} value - Value to format as currency
 * @returns {string} - Formatted currency string
 */
export const formatCurrency = (value) => {
  if (value === null || value === undefined) return '';
  
  return `$${value.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',')}`;
};

/**
 * Check if two dates are the same day
 * @param {Date} date1 - First date to compare
 * @param {Date} date2 - Second date to compare
 * @returns {boolean} - True if dates are the same day
 */
export const isSameDay = (date1, date2) => {
  if (!date1 || !date2) return false;
  
  const d1 = new Date(date1);
  const d2 = new Date(date2);
  
  return (
    d1.getFullYear() === d2.getFullYear() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getDate() === d2.getDate()
  );
};
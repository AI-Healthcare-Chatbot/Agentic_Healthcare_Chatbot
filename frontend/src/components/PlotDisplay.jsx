import React, { useState } from 'react';

const PlotDisplay = ({ plotUrl }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Construct the correct URL for the plot
  const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  // Handle any path format by extracting just the filename
  const filename = plotUrl.split(/[\\\/]/).pop();
  const fullUrl = plotUrl.startsWith('http') 
    ? plotUrl 
    : `${baseUrl}/plots/${filename}`;

  console.log("Plot URL:", plotUrl);
  console.log("Full URL:", fullUrl);

  // Toggle the expanded state
  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  // Handle image loading
  const handleImageLoad = () => {
    setIsLoading(false);
    console.log("Image loaded successfully");
  };

  // Handle image error
  const handleImageError = (e) => {
    console.error("Error loading image:", e);
    setIsLoading(false);
    setError("Failed to load plot image. Please try again.");
  };

  // Handle download
  const handleDownload = (e) => {
    e.stopPropagation();
    
    // Create an anchor element
    const a = document.createElement('a');
    a.href = fullUrl;
    // Extract filename from URL
    const filename = fullUrl.split('/').pop() || 'healthcare-plot.png';
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="mt-4">
      <div 
        className={`rounded-lg border border-gray-300 overflow-hidden ${
          isExpanded ? 'max-h-full' : 'max-h-60'
        }`}
      >
        {isLoading && (
          <div className="flex items-center justify-center h-40 bg-gray-100">
            <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
        )}
        
        {error && (
          <div className="flex items-center justify-center h-40 bg-red-50 text-red-500">
            <p>{error}</p>
          </div>
        )}
        
        <img 
          src={fullUrl} 
          alt="Healthcare data visualization" 
          className={`w-full h-auto cursor-pointer ${isLoading ? 'hidden' : 'block'}`}
          onClick={toggleExpand}
          onLoad={handleImageLoad}
          onError={handleImageError}
        />
      </div>
      
      <div className="flex mt-2 space-x-2">
        <button
          onClick={toggleExpand}
          className="px-3 py-1 text-sm text-blue-700 bg-blue-100 hover:bg-blue-200 rounded-md flex items-center"
        >
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-4 w-4 mr-1" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            {isExpanded ? (
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M4 14h5m-5 4h13a2 2 0 002-2V6a2 2 0 00-2-2H9a2 2 0 00-2 2v2m-3 4h5m-5 4h5m8-12h-5m5 0v5m0 0l-5-5" 
              />
            ) : (
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" 
              />
            )}
          </svg>
          {isExpanded ? 'Collapse' : 'Expand'}
        </button>
        
        <button
          onClick={handleDownload}
          className="px-3 py-1 text-sm text-green-700 bg-green-100 hover:bg-green-200 rounded-md flex items-center"
        >
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-4 w-4 mr-1" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4 4m0 0l-4-4m4 4V4" 
            />
          </svg>
          Download
        </button>
      </div>
    </div>
  );
};

export default PlotDisplay;
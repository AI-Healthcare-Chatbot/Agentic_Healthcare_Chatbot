import React, { useState } from 'react';
import { chatAPI } from '../services/api';

const PlotDisplay = ({ plotUrl }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Get the full URL using the API utility
  const fullUrl = chatAPI.getPlotUrl(plotUrl);

  // Toggle the expanded state
  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  // Handle image loading
  const handleImageLoad = () => {
    setIsLoading(false);
  };

  // Handle image error
  const handleImageError = () => {
    setIsLoading(false);
    setError("Failed to load visualization. Please try again.");
  };

  // Handle download
  const handleDownload = (e) => {
    e.stopPropagation();
    
    // Create an anchor element
    const a = document.createElement('a');
    a.href = fullUrl;
    a.download = plotUrl.split('/').pop() || 'healthcare-plot.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="plot-viewer">
      <div 
        className={`plot-container ${isExpanded ? 'expanded' : 'collapsed'}`}
      >
        {isLoading && (
          <div className="plot-loading">
            <div className="spinner"></div>
            <p>Loading visualization...</p>
          </div>
        )}
        
        {error && (
          <div className="plot-error">
            <p>{error}</p>
          </div>
        )}
        
        <img 
          src={fullUrl} 
          alt="Healthcare data visualization" 
          className={isLoading ? 'hidden' : 'visible'}
          onClick={toggleExpand}
          onLoad={handleImageLoad}
          onError={handleImageError}
        />
      </div>
      
      <div className="plot-controls">
        <button
          onClick={toggleExpand}
          className="control-button expand-button"
        >
          {isExpanded ? 'Collapse' : 'Expand'}
        </button>
        
        <button
          onClick={handleDownload}
          className="control-button download-button"
        >
          Download
        </button>
      </div>
    </div>
  );
};

export default PlotDisplay;
import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import PlotDisplay from './PlotDisplay';

const Message = ({ message }) => {
  const { role, content, plot } = message;
  const isUser = role === 'user';
  const [hasPlot, setHasPlot] = useState(!!plot);
  const [plotUrl, setPlotUrl] = useState(plot);

  // Check if the message contains a plot reference
  useEffect(() => {
    if (!hasPlot && content) {
      // Handle both Windows and Unix style paths
      const plotRegex = /plots[\\\/]([a-zA-Z0-9_-]+\.png)/;
      const match = plotRegex.exec(content);
      if (match) {
        console.log("Found plot reference in message:", match[1]);
        setHasPlot(true);
        
        // Set the plot URL using just the filename
        const filename = match[1];
        setPlotUrl(`/plots/${filename}`);
      }
    }
  }, [content, hasPlot]);

  // Clean content by removing plot references
  const cleanContent = content 
    ? content.replace(/Plot generated successfully: (.+\.png)/, '') 
    : '';

  return (
    <div className={`flex items-start space-x-3 ${isUser ? 'justify-end' : ''}`}>
      {/* Avatar for assistant */}
      {!isUser && (
        <div className="flex-shrink-0 bg-blue-600 rounded-full p-2">
          <svg className="w-5 h-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
          </svg>
        </div>
      )}

      {/* Message content with Markdown formatting */}
      <div className={`rounded-lg p-3 flex-1 ${
        isUser 
          ? 'bg-blue-100 text-blue-900' 
          : 'bg-gray-100 text-gray-900'
      }`}>
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]}
          components={{
            // Custom rendering for code blocks
            code({node, inline, className, children, ...props}) {
              const match = /language-(\w+)/.exec(className || '');
              return !inline && match ? (
                <SyntaxHighlighter
                  style={tomorrow}
                  language={match[1]}
                  PreTag="div"
                  {...props}
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
              ) : (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            },
            // Custom rendering for tables
            table({node, ...props}) {
              return (
                <div className="overflow-auto my-4">
                  <table className="min-w-full divide-y divide-gray-300 border border-gray-300" {...props} />
                </div>
              );
            },
            thead({node, ...props}) {
              return <thead className="bg-gray-200" {...props} />;
            },
            tbody({node, ...props}) {
              return <tbody className="divide-y divide-gray-200 bg-white" {...props} />;
            },
            tr({node, ...props}) {
              return <tr className="hover:bg-gray-50" {...props} />;
            },
            th({node, ...props}) {
              return <th className="px-4 py-3 text-left text-sm font-medium text-gray-900" {...props} />;
            },
            td({node, ...props}) {
              return <td className="px-4 py-3 text-sm text-gray-700" {...props} />;
            }
          }}
        >
          {cleanContent}
        </ReactMarkdown>

        {/* Show plot if available */}
        {(plot || (hasPlot && plotUrl)) && <PlotDisplay plotUrl={plot || plotUrl} />}
      </div>

      {/* Avatar for user */}
      {isUser && (
        <div className="flex-shrink-0 bg-blue-500 rounded-full p-2">
          <svg className="w-5 h-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
          </svg>
        </div>
      )}
    </div>
  );
};

export default Message;
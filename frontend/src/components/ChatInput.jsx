import React, { useState, useContext, useRef, useEffect } from 'react';
import { ChatContext } from '../contexts/ChatContext';

const ChatInput = () => {
  const [message, setMessage] = useState('');
  const { sendMessage, resetConversation, isLoading } = useContext(ChatContext);
  const textareaRef = useRef(null);

  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      // Reset height
      textareaRef.current.style.height = 'auto';
      // Set height to scrollHeight (content height)
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  // Handle message submission
  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      sendMessage(message);
      setMessage('');
    }
  };

  // Handle textarea input
  const handleInput = (e) => {
    setMessage(e.target.value);
  };

  // Handle keyboard shortcuts
  const handleKeyDown = (e) => {
    // Send on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Handle clear conversation
  const handleClear = () => {
    if (window.confirm('Are you sure you want to clear the conversation?')) {
      resetConversation();
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-4">
      <form onSubmit={handleSubmit} className="flex flex-col space-y-3">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask about healthcare data, diagnoses, procedures, or request a visualization..."
            rows="1"
            className="w-full px-4 py-3 pr-16 text-gray-900 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!message.trim() || isLoading}
            className={`absolute right-2 bottom-2 p-2 rounded-full text-white ${
              message.trim() && !isLoading
                ? 'bg-blue-600 hover:bg-blue-700'
                : 'bg-gray-400 cursor-not-allowed'
            }`}
          >
            <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          </button>
        </div>

        <div className="flex justify-between items-center text-xs text-gray-500">
          <div>
            <span>Press <kbd className="px-1 py-0.5 bg-gray-100 border border-gray-300 rounded">Enter</kbd> to send</span>
            <span className="mx-2">•</span>
            <span>
              Press <kbd className="px-1 py-0.5 bg-gray-100 border border-gray-300 rounded">Shift</kbd> + <kbd className="px-1 py-0.5 bg-gray-100 border border-gray-300 rounded">Enter</kbd> for a new line
            </span>
          </div>
          <button
            type="button"
            onClick={handleClear}
            className="text-red-600 hover:text-red-800 flex items-center"
          >
            <svg className="w-3.5 h-3.5 mr-1" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
            </svg>
            Clear conversation
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInput;
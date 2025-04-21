import React from 'react';
import { ChatProvider } from './contexts/ChatContext';
import Navbar from './components/Navbar';
import ChatBox from './components/ChatBox';
import ChatInput from './components/ChatInput';
import './styles/App.css';

function App() {
  return (
    <ChatProvider>
      <div className="flex flex-col h-screen bg-gray-50">
        <Navbar />
        <main className="flex-1 p-4 overflow-hidden">
          <div className="max-w-4xl mx-auto h-full flex flex-col">
            <ChatBox />
            <ChatInput />
          </div>
        </main>
      </div>
    </ChatProvider>
  );
}

export default App;
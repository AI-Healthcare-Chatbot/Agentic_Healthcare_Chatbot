import React from 'react';
import { ChatProvider } from './contexts/ChatContext';
import Header from './components/Header';
import ChatBox from './components/ChatBox';
import ChatInput from './components/ChatInput';
import './styles/App.css';

function App() {
  return (
    <ChatProvider>
      <div className="app">
        <Header />
        <div className="chat-wrapper">
          <ChatBox />
          <ChatInput />
        </div>
      </div>
    </ChatProvider>
  );
}

export default App;
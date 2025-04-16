import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Detector from './pages/Detector.tsx'; // Add .tsx extension
import './App.css';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/detector" element={<Detector />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;

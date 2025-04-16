import React from 'react';
import { Link } from 'react-router-dom';
import './Layout.css';

const Layout = ({ children }) => {
    return (
        <div className="app-container">
            <nav className="navbar">
                <div className="nav-brand">Image Forgery Detector</div>
                <div className="nav-links">
                    <Link to="/" className="nav-link">Home</Link>
                    <Link to="/detector" className="nav-link">AI Detector</Link>
                </div>
            </nav>
            <main className="main-content">
                {children}
            </main>
            <footer className="footer">
                <p>© 2024 Image Forgery Detector. All rights reserved.</p>
            </footer>
        </div>
    );
};

export default Layout;
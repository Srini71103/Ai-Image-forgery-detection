import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

const Home = () => {
    return (
        <div className="home-container">
            <div className="hero-section">
                <h1>AI-Powered Image Forgery Detection</h1>
                <p>Detect manipulated images with cutting-edge technology</p>
                <Link to="/detector" className="cta-button">Try Now</Link>
            </div>
            <div className="features-section">
                <div className="feature-card">
                    <i className="fas fa-shield-alt"></i>
                    <h3>Accurate Detection</h3>
                    <p>Advanced AI algorithms to detect image manipulation</p>
                </div>
                <div className="feature-card">
                    <i className="fas fa-bolt"></i>
                    <h3>Real-time Analysis</h3>
                    <p>Get instant results for your uploaded images</p>
                </div>
                <div className="feature-card">
                    <i className="fas fa-lock"></i>
                    <h3>Secure Process</h3>
                    <p>Your uploads are processed securely and privately</p>
                </div>
            </div>
        </div>
    );
};

export default Home;
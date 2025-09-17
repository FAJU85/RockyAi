import React from 'react';
import Logo from './Logo';
import Icon from './Icon';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-900 text-white py-8">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="col-span-1 md:col-span-2">
            <Logo size="medium" variant="full" className="mb-4" />
            <p className="text-gray-300 mb-4 max-w-md">
              Rocky AI is an advanced data analysis platform powered by artificial intelligence, 
              providing enterprise-grade analytics and insights for modern businesses.
            </p>
            <div className="flex items-center space-x-2">
              <Icon size="sm" variant="outline" />
              <span className="text-sm text-gray-400">Built for Scale • Designed for Performance</span>
            </div>
          </div>
          
          {/* Features Section */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Features</h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li>AI-Powered Analytics</li>
              <li>Real-time Processing</li>
              <li>Advanced Visualizations</li>
              <li>Enterprise Security</li>
              <li>Scalable Infrastructure</li>
              <li>API Integration</li>
            </ul>
          </div>
          
          {/* Support Section */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Support</h3>
            <ul className="space-y-2 text-sm text-gray-300">
              <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
              <li><a href="#" className="hover:text-white transition-colors">API Reference</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Community</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Status Page</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Contact Support</a></li>
            </ul>
          </div>
        </div>
        
        {/* Bottom Section */}
        <div className="border-t border-gray-800 mt-8 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-sm text-gray-400 mb-4 md:mb-0">
              © 2024 Rocky AI. All rights reserved.
            </div>
            <div className="flex space-x-6 text-sm text-gray-400">
              <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
              <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
              <a href="#" className="hover:text-white transition-colors">Security</a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

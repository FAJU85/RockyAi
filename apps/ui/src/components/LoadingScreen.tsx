import React from 'react';
import Logo from './Logo';
import Icon from './Icon';

interface LoadingScreenProps {
  message?: string;
  showProgress?: boolean;
  progress?: number;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ 
  message = "Loading Rocky AI...", 
  showProgress = false,
  progress = 0 
}) => {
  return (
    <div className="fixed inset-0 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex items-center justify-center z-50">
      <div className="text-center">
        {/* Animated Logo */}
        <div className="mb-8">
          <div className="animate-pulse">
            <Logo size="xl" className="text-white" />
          </div>
        </div>
        
        {/* Mountain Icon with Animation */}
        <div className="mb-6">
          <div className="animate-bounce">
            <Icon size="lg" variant="filled" className="text-white" />
          </div>
        </div>
        
        {/* Loading Message */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">{message}</h2>
          <p className="text-gray-300">Initializing advanced analytics platform...</p>
        </div>
        
        {/* Progress Bar */}
        {showProgress && (
          <div className="w-64 mx-auto mb-4">
            <div className="bg-gray-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-400 mt-2">{Math.round(progress)}% Complete</p>
          </div>
        )}
        
        {/* Loading Dots */}
        <div className="flex justify-center space-x-2">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
        </div>
        
        {/* Features List */}
        <div className="mt-8 text-left max-w-md mx-auto">
          <h3 className="text-lg font-semibold text-white mb-3">Powered by:</h3>
          <ul className="space-y-2 text-gray-300">
            <li className="flex items-center">
              <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
              Advanced AI Analytics
            </li>
            <li className="flex items-center">
              <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
              Real-time Data Processing
            </li>
            <li className="flex items-center">
              <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
              Enterprise Security
            </li>
            <li className="flex items-center">
              <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
              Scalable Infrastructure
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;

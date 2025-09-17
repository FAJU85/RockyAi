import React from 'react';
import Logo from './Logo';
import Icon from './Icon';

interface WelcomeScreenProps {
  onGetStarted: () => void;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onGetStarted }) => {
  const features = [
    {
      icon: '🤖',
      title: 'AI-Powered Analysis',
      description: 'Advanced machine learning algorithms for deep insights'
    },
    {
      icon: '⚡',
      title: 'Real-time Processing',
      description: 'Instant results with live data streaming'
    },
    {
      icon: '🔒',
      title: 'Enterprise Security',
      description: 'Bank-grade security and compliance'
    },
    {
      icon: '📊',
      title: 'Advanced Visualizations',
      description: 'Interactive charts and dashboards'
    },
    {
      icon: '🔧',
      title: 'Custom Workflows',
      description: 'Build and automate your analysis pipelines'
    },
    {
      icon: '📈',
      title: 'Scalable Infrastructure',
      description: 'Handle datasets of any size with ease'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 flex items-center justify-center p-6">
      <div className="max-w-6xl w-full">
        <div className="text-center mb-12">
          {/* Logo */}
          <div className="mb-8">
            <Logo size="xl" className="mx-auto" />
          </div>
          
          {/* Mountain Icon */}
          <div className="mb-6">
            <Icon size="lg" variant="filled" className="mx-auto text-gray-600" />
          </div>
          
          {/* Welcome Message */}
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Welcome to Rocky AI
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            The most advanced data analysis platform powered by artificial intelligence. 
            Transform your data into actionable insights with enterprise-grade security and performance.
          </p>
          
          {/* CTA Button */}
          <button
            onClick={onGetStarted}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 px-8 rounded-lg text-lg transition-colors shadow-lg hover:shadow-xl"
          >
            Get Started with Analysis
          </button>
        </div>
        
        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-white rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow border border-gray-200"
            >
              <div className="text-3xl mb-4">{feature.icon}</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
        
        {/* Stats Section */}
        <div className="bg-white rounded-lg p-8 shadow-md border border-gray-200">
          <h3 className="text-2xl font-bold text-gray-900 text-center mb-8">
            Trusted by Data Teams Worldwide
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-3xl font-bold text-blue-600 mb-2">10M+</div>
              <div className="text-gray-600">Analyses Completed</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-600 mb-2">500+</div>
              <div className="text-gray-600">Enterprise Clients</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-600 mb-2">99.9%</div>
              <div className="text-gray-600">Uptime Guarantee</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-600 mb-2">24/7</div>
              <div className="text-gray-600">Expert Support</div>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="text-center mt-12 text-gray-500">
          <p>Ready to unlock the power of your data?</p>
          <p className="text-sm mt-2">
            Start with a simple question or upload your dataset to begin analyzing.
          </p>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;

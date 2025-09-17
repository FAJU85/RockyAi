import React from 'react';
import { ReactComponent as LogoIcon } from '../assets/logo.svg';

interface LogoProps {
  size?: 'small' | 'medium' | 'large' | 'xl';
  variant?: 'full' | 'icon-only' | 'text-only';
  className?: string;
  onClick?: () => void;
}

const Logo: React.FC<LogoProps> = ({ 
  size = 'medium', 
  variant = 'full', 
  className = '',
  onClick 
}) => {
  const sizeClasses = {
    small: 'w-24 h-8',
    medium: 'w-32 h-10',
    large: 'w-40 h-12',
    xl: 'w-48 h-14'
  };

  const iconSizeClasses = {
    small: 'w-8 h-8',
    medium: 'w-10 h-10',
    large: 'w-12 h-12',
    xl: 'w-14 h-14'
  };

  const textSizeClasses = {
    small: 'text-sm',
    medium: 'text-lg',
    large: 'text-xl',
    xl: 'text-2xl'
  };

  const handleClick = () => {
    if (onClick) {
      onClick();
    }
  };

  if (variant === 'icon-only') {
    return (
      <div 
        className={`${iconSizeClasses[size]} ${className} ${onClick ? 'cursor-pointer hover:opacity-80 transition-opacity' : ''}`}
        onClick={handleClick}
      >
        <LogoIcon className="w-full h-full" />
      </div>
    );
  }

  if (variant === 'text-only') {
    return (
      <div 
        className={`${className} ${onClick ? 'cursor-pointer hover:opacity-80 transition-opacity' : ''}`}
        onClick={handleClick}
      >
        <span className={`font-bold text-gray-700 ${textSizeClasses[size]}`}>
          ROCKY AI
        </span>
      </div>
    );
  }

  return (
    <div 
      className={`flex items-center space-x-2 ${className} ${onClick ? 'cursor-pointer hover:opacity-80 transition-opacity' : ''}`}
      onClick={handleClick}
    >
      <div className={iconSizeClasses[size]}>
        <LogoIcon className="w-full h-full" />
      </div>
      <span className={`font-bold text-gray-700 ${textSizeClasses[size]}`}>
        ROCKY AI
      </span>
    </div>
  );
};

export default Logo;

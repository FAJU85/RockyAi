import React from 'react';
import { ReactComponent as IconSvg } from '../assets/icon.svg';

interface IconProps {
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'default' | 'outline' | 'filled';
  className?: string;
  onClick?: () => void;
  title?: string;
}

const Icon: React.FC<IconProps> = ({ 
  size = 'md', 
  variant = 'default',
  className = '',
  onClick,
  title = 'Rocky AI'
}) => {
  const sizeClasses = {
    xs: 'w-4 h-4',
    sm: 'w-6 h-6',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  };

  const variantClasses = {
    default: '',
    outline: 'border-2 border-gray-300 rounded-lg p-1',
    filled: 'bg-gray-100 rounded-lg p-1'
  };

  const handleClick = () => {
    if (onClick) {
      onClick();
    }
  };

  return (
    <div 
      className={`${sizeClasses[size]} ${variantClasses[variant]} ${className} ${onClick ? 'cursor-pointer hover:opacity-80 transition-opacity' : ''}`}
      onClick={handleClick}
      title={title}
    >
      <IconSvg className="w-full h-full" />
    </div>
  );
};

export default Icon;

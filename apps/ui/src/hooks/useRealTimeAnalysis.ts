/**
 * Real-time analysis hook
 * Manages real-time analysis updates and progress tracking
 */
import { useState, useCallback, useEffect } from 'react';
import { useWebSocket, WebSocketMessage } from './useWebSocket';

export interface AnalysisProgress {
  analysisId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number; // 0-100
  currentStep: string;
  estimatedTimeRemaining?: number;
  error?: string;
}

export interface AnalysisUpdate {
  analysisId: string;
  updateType: 'start' | 'progress' | 'complete' | 'error';
  data: any;
  timestamp: number;
}

export const useRealTimeAnalysis = (analysisId?: string) => {
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const [updates, setUpdates] = useState<AnalysisUpdate[]>([]);
  const [isSubscribed, setIsSubscribed] = useState(false);

  const wsUrl = analysisId 
    ? `ws://localhost:8000/ws/analysis/${analysisId}`
    : 'ws://localhost:8000/ws';

  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (message.type === 'analysis_update' && message.analysis_id === analysisId) {
      const update: AnalysisUpdate = {
        analysisId: message.analysis_id,
        updateType: message.update_type,
        data: message.data,
        timestamp: message.timestamp
      };

      setUpdates(prev => [...prev, update]);

      // Update progress based on update type
      switch (update.updateType) {
        case 'start':
          setProgress({
            analysisId: update.analysisId,
            status: 'running',
            progress: 0,
            currentStep: 'Starting analysis...'
          });
          break;

        case 'progress':
          setProgress(prev => prev ? {
            ...prev,
            progress: update.data.progress || prev.progress,
            currentStep: update.data.currentStep || prev.currentStep,
            estimatedTimeRemaining: update.data.estimatedTimeRemaining
          } : null);
          break;

        case 'complete':
          setProgress(prev => prev ? {
            ...prev,
            status: 'completed',
            progress: 100,
            currentStep: 'Analysis completed'
          } : null);
          break;

        case 'error':
          setProgress(prev => prev ? {
            ...prev,
            status: 'failed',
            error: update.data.error || 'Unknown error occurred'
          } : null);
          break;
      }
    } else if (message.type === 'analysis_subscribed') {
      setIsSubscribed(true);
    } else if (message.type === 'connection_established') {
      // Subscribe to analysis if analysisId is provided
      if (analysisId) {
        sendMessage({
          type: 'subscribe_analysis',
          analysis_id: analysisId
        });
      }
    }
  }, [analysisId]);

  const { sendMessage, isConnected, error } = useWebSocket({
    url: wsUrl,
    onMessage: handleMessage,
    onConnect: () => {
      if (analysisId) {
        sendMessage({
          type: 'subscribe_analysis',
          analysis_id: analysisId
        });
      }
    }
  });

  const subscribeToAnalysis = useCallback((id: string) => {
    sendMessage({
      type: 'subscribe_analysis',
      analysis_id: id
    });
  }, [sendMessage]);

  const unsubscribeFromAnalysis = useCallback((id: string) => {
    sendMessage({
      type: 'unsubscribe_analysis',
      analysis_id: id
    });
  }, [sendMessage]);

  const clearProgress = useCallback(() => {
    setProgress(null);
    setUpdates([]);
  }, []);

  return {
    progress,
    updates,
    isSubscribed,
    isConnected,
    error,
    subscribeToAnalysis,
    unsubscribeFromAnalysis,
    clearProgress
  };
};

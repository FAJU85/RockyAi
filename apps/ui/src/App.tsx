import React, { useState, useEffect } from 'react';
import { ChatPane } from './components/ChatPane';
import { DatasetPane } from './components/DatasetPane';
import { ResultsPane } from './components/ResultsPane';
import { CodeViewer } from './components/CodeViewer';
import { ProgressIndicator } from './components/ProgressIndicator';
import { useRealTimeAnalysis } from './hooks/useRealTimeAnalysis';
import Logo from './components/Logo';
import Icon from './components/Icon';
import LoadingScreen from './components/LoadingScreen';
import ErrorBoundary from './components/ErrorBoundary';
import Footer from './components/Footer';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  analysis?: {
    plan: any;
    code: string;
    status: string;
  };
}

interface Dataset {
  id: string;
  name: string;
  size: number;
  columns: string[];
  preview: any[];
  uploadedAt: Date;
}

interface AnalysisResult {
  id: string;
  query: string;
  plan: {
    analysis_type: string;
    language: string;
    steps: string[];
    libraries: string[];
  };
  code: string;
  status: 'generated' | 'executed' | 'failed';
  output?: string;
  error?: string;
  executionTime?: number;
  timestamp: Date;
}

export const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | undefined>();
  const [isLoading, setIsLoading] = useState(false);
  const [codeViewer, setCodeViewer] = useState<{ code: string; language: string } | null>(null);
  const [activePane, setActivePane] = useState<'chat' | 'datasets' | 'results'>('chat');
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [initProgress, setInitProgress] = useState(0);
  
  // Real-time analysis hook
  const { progress, isConnected: isWebSocketConnected } = useRealTimeAnalysis(currentAnalysisId);

  // Initialize app
  useEffect(() => {
    const initializeApp = async () => {
      // Simulate initialization steps
      const steps = [
        { message: "Connecting to AI services...", progress: 20 },
        { message: "Loading datasets...", progress: 40 },
        { message: "Initializing analytics engine...", progress: 60 },
        { message: "Setting up real-time connections...", progress: 80 },
        { message: "Ready to analyze!", progress: 100 }
      ];

      for (const step of steps) {
        setInitProgress(step.progress);
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      setIsInitializing(false);
    };

    initializeApp();
  }, []);

  // Load sample datasets on mount
  useEffect(() => {
    const sampleDatasets: Dataset[] = [
      {
        id: '1',
        name: 'sample_data.csv',
        size: 1024 * 1024, // 1MB
        columns: ['group', 'value', 'time'],
        preview: [
          { group: 'A', value: 10.5, time: 1 },
          { group: 'B', value: 12.3, time: 1 },
          { group: 'A', value: 11.2, time: 2 },
        ],
        uploadedAt: new Date(),
      },
    ];
    setDatasets(sampleDatasets);
  }, []);

  const handleAnalysisRequest = async (query: string) => {
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: query,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);

    setIsLoading(true);

    try {
      // Call API with enhanced error handling
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          data_path: selectedDataset?.name,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // Add assistant message with enhanced information
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: result.status === 'completed' 
          ? `✅ Analysis completed successfully! Generated ${result.plan?.analysis_type || 'an analysis'} in ${result.plan?.language || 'Python'}.`
          : result.status === 'failed'
          ? `❌ Analysis failed: ${result.error || 'Unknown error occurred'}`
          : `🔄 Generated ${result.plan?.analysis_type || 'an analysis'} for your request.`,
        timestamp: new Date(),
        analysis: {
          plan: result.plan,
          code: result.code,
          status: result.status,
        },
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Set current analysis ID for real-time updates
      if (result.analysis_id) {
        setCurrentAnalysisId(result.analysis_id);
      }

      // Add to results with enhanced metadata
      if (result.plan && result.code) {
        const analysisResult: AnalysisResult = {
          id: result.analysis_id || Date.now().toString(),
          query,
          plan: result.plan,
          code: result.code,
          status: result.status,
          output: result.output,
          error: result.error,
          executionTime: result.execution_time,
          timestamp: new Date(result.timestamp || Date.now()),
        };
        setResults(prev => [...prev, analysisResult]);
      }

    } catch (error) {
      console.error('Analysis request failed:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `❌ Sorry, I encountered an error processing your request: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDatasetUpload = async (file: File) => {
    // Simulate file upload
    const newDataset: Dataset = {
      id: Date.now().toString(),
      name: file.name,
      size: file.size,
      columns: ['column1', 'column2', 'column3'], // Mock columns
      preview: [
        { column1: 'value1', column2: 'value2', column3: 'value3' },
        { column1: 'value4', column2: 'value5', column3: 'value6' },
      ],
      uploadedAt: new Date(),
    };
    
    setDatasets(prev => [...prev, newDataset]);
  };

  const handleCodeView = (code: string) => {
    setCodeViewer({ code, language: 'python' });
  };

  // Show loading screen during initialization
  if (isInitializing) {
    return <LoadingScreen message="Initializing Rocky AI..." showProgress progress={initProgress} />;
  }

  return (
    <ErrorBoundary>
      <div className="h-screen bg-gray-100 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Logo size="large" onClick={() => setActivePane('chat')} className="hover:opacity-80 transition-opacity" />
            <span className="text-sm text-gray-500">Advanced Data Analysis Platform</span>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Icon size="sm" variant="outline" />
              <span className="text-sm text-gray-600">AI-Powered Analytics</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isWebSocketConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">{isWebSocketConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <div className="w-80 flex-shrink-0">
          <div className="h-full flex flex-col">
            {/* Pane Selector */}
            <div className="bg-white border-b border-gray-200">
              <nav className="flex">
                {[
                  { id: 'chat', label: 'Chat', icon: '💬', description: 'AI Assistant' },
                  { id: 'datasets', label: 'Data', icon: '📊', description: 'Datasets' },
                  { id: 'results', label: 'Results', icon: '📈', description: 'Analytics' },
                ].map((pane) => (
                  <button
                    key={pane.id}
                    onClick={() => setActivePane(pane.id as any)}
                    className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                      activePane === pane.id
                        ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                        : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                    }`}
                    title={pane.description}
                  >
                    <div className="flex flex-col items-center space-y-1">
                      <span className="text-lg">{pane.icon}</span>
                      <span>{pane.label}</span>
                    </div>
                  </button>
                ))}
              </nav>
            </div>

            {/* Pane Content */}
            <div className="flex-1">
              {activePane === 'chat' && (
                <ChatPane
                  onAnalysisRequest={handleAnalysisRequest}
                  messages={messages}
                  isLoading={isLoading}
                />
              )}
              {activePane === 'datasets' && (
                <DatasetPane
                  datasets={datasets}
                  onDatasetSelect={setSelectedDataset}
                  onDatasetUpload={handleDatasetUpload}
                  selectedDataset={selectedDataset}
                />
              )}
              {activePane === 'results' && (
                <ResultsPane
                  results={results}
                  onCodeView={handleCodeView}
                />
              )}
            </div>

            {/* Real-time Progress Indicator */}
            {progress && (
              <div className="p-4 border-t border-gray-200 bg-gray-50">
                <ProgressIndicator progress={progress} />
              </div>
            )}

            {/* WebSocket Connection Status */}
            <div className="px-4 py-2 border-t border-gray-200 bg-gray-50">
              <div className="flex items-center space-x-2 text-sm">
                <div className={`w-2 h-2 rounded-full ${
                  isWebSocketConnected ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-gray-600">
                  {isWebSocketConnected ? 'Real-time updates connected' : 'Real-time updates disconnected'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Content Area */}
        <div className="flex-1 bg-white">
          {activePane === 'results' ? (
            <div className="h-full flex items-center justify-center text-gray-500">
              <div className="text-center">
                <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Analysis Results</h3>
                <p className="text-sm text-gray-600">
                  Select a result from the sidebar to view details
                </p>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-500">
              <div className="text-center">
                <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Welcome to Rocky AI</h3>
                <p className="text-sm text-gray-600">
                  Start by asking questions about your data or uploading a dataset
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Code Viewer Modal */}
      {codeViewer && (
        <CodeViewer
          code={codeViewer.code}
          language={codeViewer.language}
          onClose={() => setCodeViewer(null)}
        />
      )}
      
      {/* Footer */}
      <Footer />
      </div>
    </ErrorBoundary>
  );
};
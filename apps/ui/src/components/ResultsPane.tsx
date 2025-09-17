import React, { useState } from 'react';

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
  status: 'generated' | 'executed' | 'completed' | 'failed';
  output?: string;
  error?: string;
  executionTime?: number;
  timestamp: Date;
}

interface ResultsPaneProps {
  results: AnalysisResult[];
  onCodeView: (code: string) => void;
}

export const ResultsPane: React.FC<ResultsPaneProps> = ({ results, onCodeView }) => {
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<'output' | 'code' | 'plan'>('output');

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'generated':
        return 'bg-yellow-100 text-yellow-800';
      case 'executed':
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'generated':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'executed':
      case 'completed':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        );
      case 'failed':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex h-full bg-white">
      {/* Results List */}
      <div className="w-1/3 border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <h2 className="text-lg font-semibold text-gray-800">Analysis Results</h2>
          <p className="text-sm text-gray-600">{results.length} analyses completed</p>
        </div>

        <div className="flex-1 overflow-y-auto">
          {results.length === 0 ? (
            <div className="p-4 text-center text-gray-500">
              <div className="mb-4">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No results yet</h3>
              <p className="text-sm text-gray-600">
                Start an analysis to see results here
              </p>
            </div>
          ) : (
            <div className="p-2 space-y-2">
              {results.map((result) => (
                <div
                  key={result.id}
                  onClick={() => setSelectedResult(result)}
                  className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                    selectedResult?.id === result.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="text-sm font-medium text-gray-900 truncate">
                        {result.query}
                      </h3>
                      <div className="mt-1 flex items-center space-x-2">
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(result.status)}`}>
                          {getStatusIcon(result.status)}
                          <span className="ml-1">{result.status}</span>
                        </span>
                        <span className="text-xs text-gray-500">
                          {result.plan.analysis_type}
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-gray-400">
                        {result.timestamp.toLocaleString()}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Result Details */}
      <div className="flex-1 flex flex-col">
        {selectedResult ? (
          <>
            {/* Header */}
            <div className="p-4 border-b border-gray-200 bg-gray-50">
              <h3 className="text-lg font-semibold text-gray-800 truncate">
                {selectedResult.query}
              </h3>
              <div className="mt-2 flex items-center space-x-4">
                <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedResult.status)}`}>
                  {getStatusIcon(selectedResult.status)}
                  <span className="ml-1">{selectedResult.status}</span>
                </span>
                <span className="text-sm text-gray-600">
                  {selectedResult.plan.language.toUpperCase()} • {selectedResult.plan.analysis_type}
                </span>
                {selectedResult.executionTime && (
                  <span className="text-sm text-gray-600">
                    {selectedResult.executionTime.toFixed(2)}s
                  </span>
                )}
              </div>
            </div>

            {/* Tabs */}
            <div className="border-b border-gray-200">
              <nav className="flex space-x-8 px-4">
                {['output', 'code', 'plan'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab as any)}
                    className={`py-4 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </nav>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {activeTab === 'output' && (
                <div className="space-y-4">
                  {selectedResult.error ? (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                      <div className="flex">
                        <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <div className="ml-3">
                          <h3 className="text-sm font-medium text-red-800">Error</h3>
                          <div className="mt-2 text-sm text-red-700">
                            <pre className="whitespace-pre-wrap">{selectedResult.error}</pre>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : selectedResult.output ? (
                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                      <h4 className="text-sm font-medium text-gray-800 mb-2">Analysis Output</h4>
                      <pre className="text-sm text-gray-700 whitespace-pre-wrap overflow-x-auto">
                        {selectedResult.output}
                      </pre>
                    </div>
                  ) : (
                    <div className="text-center text-gray-500 py-8">
                      <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <p>No output available yet</p>
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'code' && (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h4 className="text-sm font-medium text-gray-800">Generated Code</h4>
                    <button
                      onClick={() => onCodeView(selectedResult.code)}
                      className="px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      View Full Code
                    </button>
                  </div>
                  <div className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm">
                      <code>{selectedResult.code}</code>
                    </pre>
                  </div>
                </div>
              )}

              {activeTab === 'plan' && (
                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-gray-800">Analysis Plan</h4>
                  
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Analysis Type</label>
                        <p className="mt-1 text-sm text-gray-900">{selectedResult.plan.analysis_type}</p>
                      </div>
                      <div>
                        <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Language</label>
                        <p className="mt-1 text-sm text-gray-900">{selectedResult.plan.language.toUpperCase()}</p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Steps</label>
                    <ol className="mt-2 space-y-2">
                      {selectedResult.plan.steps.map((step, index) => (
                        <li key={index} className="flex items-start">
                          <span className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xs font-medium">
                            {index + 1}
                          </span>
                          <span className="ml-3 text-sm text-gray-900">{step}</span>
                        </li>
                      ))}
                    </ol>
                  </div>

                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    <label className="text-xs font-medium text-gray-500 uppercase tracking-wide">Required Libraries</label>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {selectedResult.plan.libraries.map((library, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded"
                        >
                          {library}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <h3 className="text-lg font-medium text-gray-900 mb-2">Select a result</h3>
              <p className="text-sm text-gray-600">
                Choose an analysis result to view details
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

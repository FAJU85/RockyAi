import React, { useState, useRef } from 'react';

interface Dataset {
  id: string;
  name: string;
  size: number;
  columns: string[];
  preview: any[];
  uploadedAt: Date;
}

interface DatasetPaneProps {
  datasets: Dataset[];
  onDatasetSelect: (dataset: Dataset) => void;
  onDatasetUpload: (file: File) => void;
  selectedDataset?: Dataset;
}

export const DatasetPane: React.FC<DatasetPaneProps> = ({
  datasets,
  onDatasetSelect,
  onDatasetUpload,
  selectedDataset
}) => {
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setIsUploading(true);
      try {
        await onDatasetUpload(file);
      } finally {
        setIsUploading(false);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="flex flex-col h-full bg-white border-r border-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <h2 className="text-lg font-semibold text-gray-800">Datasets</h2>
        <p className="text-sm text-gray-600">Upload and manage your data files</p>
      </div>

      {/* Upload Section */}
      <div className="p-4 border-b border-gray-200">
        <div className="space-y-3">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            {isUploading ? 'Uploading...' : 'Upload Dataset'}
          </button>
          
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.parquet,.xlsx,.json"
            onChange={handleFileUpload}
            className="hidden"
          />
          
          <div className="text-xs text-gray-500">
            Supported formats: CSV, Parquet, Excel, JSON
          </div>
        </div>
      </div>

      {/* Dataset List */}
      <div className="flex-1 overflow-y-auto">
        {datasets.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            <div className="mb-4">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No datasets uploaded</h3>
            <p className="text-sm text-gray-600">
              Upload a dataset to get started with your analysis
            </p>
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                onClick={() => onDatasetSelect(dataset)}
                className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                  selectedDataset?.id === dataset.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium text-gray-900 truncate">
                      {dataset.name}
                    </h3>
                    <div className="mt-1 text-xs text-gray-500">
                      {dataset.columns.length} columns • {formatFileSize(dataset.size)}
                    </div>
                    <div className="mt-1 text-xs text-gray-400">
                      Uploaded {dataset.uploadedAt.toLocaleDateString()}
                    </div>
                  </div>
                  <div className="ml-2 flex-shrink-0">
                    <svg className="h-4 w-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Selected Dataset Details */}
      {selectedDataset && (
        <div className="p-4 border-t border-gray-200 bg-gray-50">
          <h3 className="text-sm font-medium text-gray-900 mb-2">Dataset Preview</h3>
          <div className="text-xs text-gray-600 mb-2">
            {selectedDataset.columns.length} columns, {selectedDataset.preview.length} rows shown
          </div>
          
          {/* Column List */}
          <div className="mb-3">
            <div className="text-xs font-medium text-gray-700 mb-1">Columns:</div>
            <div className="flex flex-wrap gap-1">
              {selectedDataset.columns.map((column, index) => (
                <span
                  key={index}
                  className="px-2 py-1 bg-white text-xs text-gray-600 rounded border"
                >
                  {column}
                </span>
              ))}
            </div>
          </div>

          {/* Data Preview */}
          <div className="max-h-32 overflow-y-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-200">
                  {selectedDataset.columns.slice(0, 5).map((column, index) => (
                    <th key={index} className="text-left py-1 px-2 font-medium text-gray-700">
                      {column}
                    </th>
                  ))}
                  {selectedDataset.columns.length > 5 && (
                    <th className="text-left py-1 px-2 font-medium text-gray-700">
                      ...
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {selectedDataset.preview.slice(0, 3).map((row, rowIndex) => (
                  <tr key={rowIndex} className="border-b border-gray-100">
                    {selectedDataset.columns.slice(0, 5).map((column, colIndex) => (
                      <td key={colIndex} className="py-1 px-2 text-gray-600">
                        {row[column]?.toString().substring(0, 20) || '-'}
                        {row[column]?.toString().length > 20 && '...'}
                      </td>
                    ))}
                    {selectedDataset.columns.length > 5 && (
                      <td className="py-1 px-2 text-gray-400">...</td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

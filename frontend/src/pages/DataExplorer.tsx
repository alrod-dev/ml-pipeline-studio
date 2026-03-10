import React, { useEffect, useState } from 'react'
import { datasets } from '../lib/api'
import { usePipelineStore } from '../store/pipelineStore'
import DatasetStats from '../components/DatasetStats'
import type { Dataset } from '../types'

const DataExplorer: React.FC = () => {
  const { datasets: storeDatasets, addDataset, setLoading, loading, error, setError } = usePipelineStore()
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [preview, setPreview] = useState<any>(null)

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setLoading(true)
    try {
      const newDataset = await datasets.upload(file)
      addDataset(newDataset)
      setSelectedDataset(newDataset)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  const loadPreview = async (dataset: Dataset) => {
    try {
      const previewData = await datasets.preview(dataset.id, 10)
      setPreview(previewData)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load preview')
    }
  }

  useEffect(() => {
    if (selectedDataset) {
      loadPreview(selectedDataset)
    }
  }, [selectedDataset])

  return (
    <div className="space-y-8">
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Data Explorer</h2>

        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input
            type="file"
            id="file-input"
            accept=".csv,.json,.xlsx"
            onChange={handleFileUpload}
            disabled={loading}
            className="hidden"
          />
          <label htmlFor="file-input" className="cursor-pointer">
            <p className="text-lg font-semibold text-gray-700 mb-2">
              {loading ? 'Uploading...' : 'Drag and drop or click to upload'}
            </p>
            <p className="text-sm text-gray-500">Supported formats: CSV, JSON, XLSX</p>
          </label>
        </div>

        {error && (
          <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}
      </div>

      {storeDatasets.length > 0 && (
        <div className="card">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Available Datasets</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {storeDatasets.map((dataset) => (
              <div
                key={dataset.id}
                onClick={() => setSelectedDataset(dataset)}
                className={`p-4 rounded-lg border-2 cursor-pointer transition ${
                  selectedDataset?.id === dataset.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-300 hover:border-blue-300'
                }`}
              >
                <p className="font-semibold text-gray-800">{dataset.name}</p>
                <p className="text-sm text-gray-600">{dataset.rows} rows × {dataset.columns} columns</p>
                <p className="text-xs text-gray-500">{dataset.file_size_mb.toFixed(2)} MB</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedDataset && (
        <>
          <DatasetStats datasetId={selectedDataset.id} />

          {preview && (
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">Data Preview</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-200">
                    <tr>
                      {preview.columns.map((col: string) => (
                        <th key={col} className="px-4 py-2 text-left font-semibold text-gray-800">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row: any, idx: number) => (
                      <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        {preview.columns.map((col: string) => (
                          <td key={`${idx}-${col}`} className="px-4 py-2 text-gray-800">
                            {String(row[col] ?? '').substring(0, 50)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default DataExplorer

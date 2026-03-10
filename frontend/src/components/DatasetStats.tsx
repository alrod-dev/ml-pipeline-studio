import React, { useEffect, useState } from 'react'
import { datasets } from '../lib/api'
import type { DatasetStats as DatasetStatsType } from '../types'

interface DatasetStatsProps {
  datasetId: string
}

const DatasetStats: React.FC<DatasetStatsProps> = ({ datasetId }) => {
  const [stats, setStats] = useState<DatasetStatsType | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await datasets.getStats(datasetId)
        setStats(data)
      } catch (err) {
        console.error('Failed to load dataset stats')
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
  }, [datasetId])

  if (loading) {
    return <div className="text-center py-8">Loading statistics...</div>
  }

  if (!stats) {
    return <div className="text-center py-8">Failed to load statistics</div>
  }

  return (
    <div className="card space-y-6">
      <h3 className="text-xl font-semibold text-gray-800">Dataset Statistics</h3>

      {/* Overview */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Total Rows</p>
          <p className="text-2xl font-bold text-blue-600">{stats.rows}</p>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Columns</p>
          <p className="text-2xl font-bold text-green-600">{stats.columns}</p>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Missing Values</p>
          <p className="text-2xl font-bold text-purple-600">
            {Object.values(stats.missing_values).reduce((a, b) => a + b, 0)}
          </p>
        </div>
      </div>

      {/* Numeric Columns Stats */}
      {stats.numeric_columns.length > 0 && (
        <div>
          <h4 className="font-semibold text-gray-800 mb-3">Numeric Columns</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {stats.numeric_columns.map((col) => {
              const colStats = stats.numeric_stats[col]
              return (
                <div key={col} className="border rounded-lg p-4">
                  <p className="font-semibold text-gray-800 mb-3">{col}</p>
                  <div className="space-y-1 text-sm">
                    <p className="flex justify-between text-gray-700">
                      <span>Mean:</span>
                      <span className="font-medium">{colStats?.mean?.toFixed(2) ?? 'N/A'}</span>
                    </p>
                    <p className="flex justify-between text-gray-700">
                      <span>Median:</span>
                      <span className="font-medium">{colStats?.median?.toFixed(2) ?? 'N/A'}</span>
                    </p>
                    <p className="flex justify-between text-gray-700">
                      <span>Std Dev:</span>
                      <span className="font-medium">{colStats?.std?.toFixed(2) ?? 'N/A'}</span>
                    </p>
                    <p className="flex justify-between text-gray-700">
                      <span>Min:</span>
                      <span className="font-medium">{colStats?.min?.toFixed(2) ?? 'N/A'}</span>
                    </p>
                    <p className="flex justify-between text-gray-700">
                      <span>Max:</span>
                      <span className="font-medium">{colStats?.max?.toFixed(2) ?? 'N/A'}</span>
                    </p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Categorical Columns */}
      {stats.categorical_columns.length > 0 && (
        <div>
          <h4 className="font-semibold text-gray-800 mb-3">Categorical Columns</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {stats.categorical_columns.map((col) => (
              <div key={col} className="border rounded-lg p-4">
                <p className="font-semibold text-gray-800 mb-2">{col}</p>
                <p className="text-sm text-gray-600">
                  Unique values: <span className="font-medium">{stats.missing_values[col] ?? 0}</span>
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default DatasetStats

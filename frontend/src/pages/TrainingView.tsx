import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { pipelines } from '../lib/api'
import type { Pipeline } from '../types'

const TrainingView: React.FC = () => {
  const { pipelineId } = useParams<{ pipelineId: string }>()
  const [pipeline, setPipeline] = useState<Pipeline | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchPipeline = async () => {
      if (!pipelineId) return

      try {
        const result = await pipelines.get(pipelineId)
        setPipeline(result)

        // Poll for updates if still running
        if (result.status === 'running') {
          const interval = setInterval(async () => {
            const updated = await pipelines.get(pipelineId)
            setPipeline(updated)
            if (updated.status !== 'running') {
              clearInterval(interval)
            }
          }, 2000)

          return () => clearInterval(interval)
        }
      } catch (err) {
        console.error('Failed to load pipeline')
      } finally {
        setLoading(false)
      }
    }

    fetchPipeline()
  }, [pipelineId])

  if (loading) {
    return <div className="text-center py-12">Loading pipeline...</div>
  }

  if (!pipeline) {
    return <div className="text-center py-12">Pipeline not found</div>
  }

  return (
    <div className="space-y-8">
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Pipeline Results</h2>
        <p className="text-gray-600">ID: {pipeline.pipeline_id}</p>

        <div className="mt-4 p-4 rounded-lg bg-gray-50">
          <p className="text-sm text-gray-600">Status</p>
          <span
            className={`inline-block mt-1 px-3 py-1 rounded-lg font-semibold ${
              pipeline.status === 'completed'
                ? 'bg-green-100 text-green-800'
                : pipeline.status === 'failed'
                ? 'bg-red-100 text-red-800'
                : 'bg-yellow-100 text-yellow-800'
            }`}
          >
            {pipeline.status.toUpperCase()}
          </span>
        </div>

        {pipeline.error && (
          <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {pipeline.error}
          </div>
        )}
      </div>

      {pipeline.metrics && (
        <div className="card">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Metrics</h3>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {pipeline.metrics.accuracy !== undefined && (
              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">Accuracy</p>
                <p className="text-2xl font-bold text-blue-600">
                  {(pipeline.metrics.accuracy * 100).toFixed(2)}%
                </p>
              </div>
            )}
            {pipeline.metrics.f1 !== undefined && (
              <div className="bg-green-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">F1 Score</p>
                <p className="text-2xl font-bold text-green-600">
                  {(pipeline.metrics.f1 * 100).toFixed(2)}%
                </p>
              </div>
            )}
            {pipeline.metrics.r2 !== undefined && (
              <div className="bg-purple-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600">R² Score</p>
                <p className="text-2xl font-bold text-purple-600">
                  {(pipeline.metrics.r2 * 100).toFixed(2)}%
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {pipeline.feature_importance && (
        <div className="card">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Top Features</h3>
          <div className="space-y-2">
            {Object.entries(pipeline.feature_importance)
              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .slice(0, 10)
              .map(([feature, importance]) => (
                <div key={feature} className="flex items-center justify-between">
                  <p className="text-gray-800">{feature}</p>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${Math.abs(importance) * 100}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-600 w-16 text-right">
                      {(importance * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {pipeline.charts && Object.keys(pipeline.charts).length > 0 && (
        <div className="card">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Visualizations</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(pipeline.charts).map(([name, chart]) => (
              <div key={name} className="border rounded-lg overflow-hidden">
                <img src={chart} alt={name} className="w-full h-auto" />
                <p className="text-sm text-gray-600 p-2 capitalize">{name.replace('_', ' ')}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default TrainingView

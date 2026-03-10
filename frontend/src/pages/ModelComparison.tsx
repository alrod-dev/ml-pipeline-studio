import React, { useEffect } from 'react'
import { models } from '../lib/api'
import { usePipelineStore } from '../store/pipelineStore'

const ModelComparison: React.FC = () => {
  const { models: storeModels, setModels, loading, setLoading } = usePipelineStore()

  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true)
      try {
        const list = await models.list()
        setModels(list)
      } catch (err) {
        console.error('Failed to load models')
      } finally {
        setLoading(false)
      }
    }

    fetchModels()
  }, [setModels, setLoading])

  if (loading) {
    return <div className="text-center py-12">Loading models...</div>
  }

  if (storeModels.length === 0) {
    return (
      <div className="card text-center py-12">
        <p className="text-gray-600">No trained models yet. Create a pipeline to train models.</p>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Trained Models</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {storeModels.map((model) => (
            <div key={model.id} className="border rounded-lg p-6 hover:shadow-lg transition">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">{model.name || model.id}</h3>

              <div className="space-y-2 text-sm text-gray-600 mb-4">
                <p>
                  <span className="font-semibold">Type:</span> {model.model_type}
                </p>
                <p>
                  <span className="font-semibold">Problem:</span> {model.problem_type}
                </p>
                <p>
                  <span className="font-semibold">Created:</span>{' '}
                  {new Date(model.created_at).toLocaleDateString()}
                </p>
              </div>

              <div className="space-y-2">
                {model.metrics.accuracy !== undefined && (
                  <div>
                    <p className="text-sm font-semibold text-gray-700">Accuracy</p>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${model.metrics.accuracy * 100}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      {(model.metrics.accuracy * 100).toFixed(2)}%
                    </p>
                  </div>
                )}

                {model.metrics.r2 !== undefined && (
                  <div>
                    <p className="text-sm font-semibold text-gray-700">R² Score</p>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-600 h-2 rounded-full"
                        style={{ width: `${Math.max(0, model.metrics.r2 * 100)}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      {(model.metrics.r2 * 100).toFixed(2)}%
                    </p>
                  </div>
                )}
              </div>

              <button
                className="button-primary mt-4 w-full"
                onClick={() => window.alert(`Model ${model.id} details would open here`)}
              >
                View Details
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Comparison Table */}
      <div className="card">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Comparison</h3>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-200">
              <tr>
                <th className="px-4 py-2 text-left font-semibold">Model</th>
                <th className="px-4 py-2 text-left font-semibold">Type</th>
                <th className="px-4 py-2 text-right font-semibold">Accuracy</th>
                <th className="px-4 py-2 text-right font-semibold">F1</th>
                <th className="px-4 py-2 text-right font-semibold">R²</th>
              </tr>
            </thead>
            <tbody>
              {storeModels.map((model) => (
                <tr key={model.id} className="border-t hover:bg-gray-50">
                  <td className="px-4 py-2 text-gray-800">{model.name || model.id}</td>
                  <td className="px-4 py-2 text-gray-600">{model.model_type}</td>
                  <td className="px-4 py-2 text-right text-gray-800">
                    {model.metrics.accuracy
                      ? (model.metrics.accuracy * 100).toFixed(2) + '%'
                      : '-'}
                  </td>
                  <td className="px-4 py-2 text-right text-gray-800">
                    {model.metrics.f1 ? (model.metrics.f1 * 100).toFixed(2) + '%' : '-'}
                  </td>
                  <td className="px-4 py-2 text-right text-gray-800">
                    {model.metrics.r2 ? (model.metrics.r2 * 100).toFixed(2) + '%' : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default ModelComparison

import React, { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { datasets, pipelines, models } from '../lib/api'
import { usePipelineStore } from '../store/pipelineStore'

const Dashboard: React.FC = () => {
  const {
    datasets: storeDatasets,
    models: storeModels,
    pipelines: storePipelines,
    setDatasets,
    setModels,
    setPipelines,
    loading,
    setLoading,
    error,
    setError,
  } = usePipelineStore()

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      try {
        const [datasetsList, modelsList, pipelinesList] = await Promise.all([
          datasets.list(),
          models.list(),
          pipelines.list(),
        ])
        setDatasets(datasetsList)
        setModels(modelsList)
        setPipelines(pipelinesList)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [setDatasets, setModels, setPipelines, setLoading, setError])

  if (loading) {
    return <div className="text-center py-12">Loading...</div>
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-3 gap-6">
        <div className="card">
          <h3 className="text-gray-600 text-sm font-semibold">Datasets</h3>
          <p className="text-4xl font-bold text-blue-600">{storeDatasets.length}</p>
        </div>
        <div className="card">
          <h3 className="text-gray-600 text-sm font-semibold">Models Trained</h3>
          <p className="text-4xl font-bold text-green-600">{storeModels.length}</p>
        </div>
        <div className="card">
          <h3 className="text-gray-600 text-sm font-semibold">Pipelines Run</h3>
          <p className="text-4xl font-bold text-purple-600">{storePipelines.length}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <Link
          to="/data-explorer"
          className="card hover:shadow-lg transition cursor-pointer"
        >
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Upload & Explore Data</h3>
          <p className="text-gray-600">Load datasets and analyze their properties</p>
        </Link>

        <Link
          to="/pipeline-builder"
          className="card hover:shadow-lg transition cursor-pointer"
        >
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Build Pipeline</h3>
          <p className="text-gray-600">Create and configure ML pipelines</p>
        </Link>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent Pipelines</h3>
        {storePipelines.length === 0 ? (
          <p className="text-gray-600">No pipelines have been run yet</p>
        ) : (
          <div className="space-y-2">
            {storePipelines.slice(0, 5).map((pipeline) => (
              <div key={pipeline.pipeline_id} className="p-3 bg-gray-50 rounded flex justify-between items-center">
                <div>
                  <p className="font-medium text-gray-800">{pipeline.pipeline_id}</p>
                  <p className="text-sm text-gray-600">{new Date(pipeline.timestamp).toLocaleDateString()}</p>
                </div>
                <span className={`px-3 py-1 rounded text-sm font-medium ${
                  pipeline.status === 'completed' ? 'bg-green-100 text-green-800' :
                  pipeline.status === 'failed' ? 'bg-red-100 text-red-800' :
                  'bg-yellow-100 text-yellow-800'
                }`}>
                  {pipeline.status}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard

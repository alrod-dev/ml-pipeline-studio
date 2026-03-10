import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { datasets, pipelines } from '../lib/api'
import { usePipelineStore } from '../store/pipelineStore'
import type { Dataset, PipelineConfig } from '../types'

const PipelineBuilder: React.FC = () => {
  const navigate = useNavigate()
  const { addPipeline, setLoading, loading } = usePipelineStore()
  const [datasetsList, setDatasetsList] = useState<Dataset[]>([])
  const [config, setConfig] = useState<Partial<PipelineConfig>>({
    problem_type: 'classification',
    model_type: 'random_forest',
    preprocessing: {
      scaling: { type: 'standard' },
      encoding: { type: 'label' },
    },
  })

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const list = await datasets.list()
        setDatasetsList(list)
      } catch (err) {
        console.error('Failed to load datasets')
      }
    }

    fetchDatasets()
  }, [])

  const handleConfigChange = (key: keyof PipelineConfig, value: any) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
  }

  const handleRunPipeline = async () => {
    if (!config.dataset_id || !config.target_column || !config.model_type) {
      alert('Please fill in all required fields')
      return
    }

    setLoading(true)
    try {
      const result = await pipelines.run(config as PipelineConfig)
      addPipeline(result)

      if (result.status === 'completed') {
        navigate(`/training/${result.pipeline_id}`)
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Pipeline execution failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">ML Pipeline Configuration</h2>

        <div className="space-y-6">
          {/* Dataset Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Select Dataset
            </label>
            <select
              value={config.dataset_id || ''}
              onChange={(e) => handleConfigChange('dataset_id', e.target.value)}
              className="input-base"
            >
              <option value="">Choose a dataset...</option>
              {datasetsList.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.name} ({ds.rows} rows)
                </option>
              ))}
            </select>
          </div>

          {/* Target Column */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Target Column
            </label>
            <input
              type="text"
              value={config.target_column || ''}
              onChange={(e) => handleConfigChange('target_column', e.target.value)}
              placeholder="e.g., target, label, class"
              className="input-base"
            />
          </div>

          {/* Problem Type */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Problem Type
            </label>
            <select
              value={config.problem_type || ''}
              onChange={(e) => handleConfigChange('problem_type', e.target.value as any)}
              className="input-base"
            >
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
              <option value="clustering">Clustering</option>
            </select>
          </div>

          {/* Model Type */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Model Type
            </label>
            <select
              value={config.model_type || ''}
              onChange={(e) => handleConfigChange('model_type', e.target.value)}
              className="input-base"
            >
              {config.problem_type === 'classification' && (
                <>
                  <option value="random_forest">Random Forest</option>
                  <option value="gradient_boosting">Gradient Boosting</option>
                  <option value="svm">Support Vector Machine</option>
                  <option value="logistic_regression">Logistic Regression</option>
                </>
              )}
              {config.problem_type === 'regression' && (
                <>
                  <option value="linear_regression">Linear Regression</option>
                  <option value="random_forest">Random Forest</option>
                  <option value="gradient_boosting">Gradient Boosting</option>
                  <option value="svm">Support Vector Machine</option>
                </>
              )}
              {config.problem_type === 'clustering' && (
                <option value="kmeans">K-Means</option>
              )}
            </select>
          </div>

          {/* Preprocessing Options */}
          <div className="border-t pt-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Preprocessing</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Feature Scaling
                </label>
                <select
                  value={config.preprocessing?.scaling?.type || 'standard'}
                  onChange={(e) =>
                    handleConfigChange('preprocessing', {
                      ...config.preprocessing,
                      scaling: { type: e.target.value },
                    })
                  }
                  className="input-base"
                >
                  <option value="standard">Standard Scaler</option>
                  <option value="minmax">MinMax Scaler</option>
                  <option value="robust">Robust Scaler</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Categorical Encoding
                </label>
                <select
                  value={config.preprocessing?.encoding?.type || 'label'}
                  onChange={(e) =>
                    handleConfigChange('preprocessing', {
                      ...config.preprocessing,
                      encoding: { type: e.target.value, columns: [] },
                    })
                  }
                  className="input-base"
                >
                  <option value="label">Label Encoding</option>
                  <option value="onehot">One-Hot Encoding</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleRunPipeline}
          disabled={loading}
          className="button-primary mt-8 w-full"
        >
          {loading ? 'Running Pipeline...' : 'Run Pipeline'}
        </button>
      </div>
    </div>
  )
}

export default PipelineBuilder

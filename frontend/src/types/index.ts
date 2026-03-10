export interface Dataset {
  id: string
  name: string
  filename: string
  rows: number
  columns: number
  created_at: string
  file_size_mb: number
}

export interface DatasetStats {
  id: string
  name: string
  rows: number
  columns: number
  numeric_columns: string[]
  categorical_columns: string[]
  missing_values: Record<string, number>
  numeric_stats: Record<string, Record<string, number>>
}

export interface Model {
  id: string
  name: string
  model_type: string
  problem_type: string
  dataset_id: string
  metrics: ModelMetrics
  created_at: string
  charts?: Record<string, string>
}

export interface ModelMetrics {
  accuracy?: number
  f1?: number
  precision?: number
  recall?: number
  roc_auc?: number
  mse?: number
  rmse?: number
  mae?: number
  r2?: number
  confusion_matrix?: number[][]
  classification_report?: Record<string, any>
}

export interface Pipeline {
  pipeline_id: string
  status: 'running' | 'completed' | 'failed'
  model_id?: string
  metrics?: ModelMetrics
  charts?: Record<string, string>
  error?: string
  timestamp: string
  feature_importance?: Record<string, number>
  test_samples_count?: number
  train_samples_count?: number
}

export interface PipelineConfig {
  dataset_id: string
  target_column: string
  model_type: string
  problem_type: 'classification' | 'regression' | 'clustering'
  preprocessing?: Record<string, any>
  hyperparameters?: Record<string, any>
}

export interface PipelineStep {
  id: string
  name: string
  type: 'preprocessing' | 'training' | 'evaluation'
  config: Record<string, any>
}

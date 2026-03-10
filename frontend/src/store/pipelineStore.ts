import { create } from 'zustand'
import type { Dataset, Pipeline, Model, PipelineConfig } from '../types'

interface PipelineStore {
  datasets: Dataset[]
  models: Model[]
  pipelines: Pipeline[]
  selectedDataset: Dataset | null
  selectedModel: Model | null
  loading: boolean
  error: string | null

  // Dataset actions
  setDatasets: (datasets: Dataset[]) => void
  addDataset: (dataset: Dataset) => void
  removeDataset: (id: string) => void
  selectDataset: (dataset: Dataset | null) => void

  // Model actions
  setModels: (models: Model[]) => void
  addModel: (model: Model) => void
  removeModel: (id: string) => void
  selectModel: (model: Model | null) => void

  // Pipeline actions
  setPipelines: (pipelines: Pipeline[]) => void
  addPipeline: (pipeline: Pipeline) => void
  updatePipeline: (id: string, pipeline: Pipeline) => void

  // State management
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void

  // UI state
  currentStep: 'data' | 'pipeline' | 'training' | 'results'
  setCurrentStep: (step: 'data' | 'pipeline' | 'training' | 'results') => void

  pipelineConfig: Partial<PipelineConfig>
  updatePipelineConfig: (config: Partial<PipelineConfig>) => void
}

export const usePipelineStore = create<PipelineStore>((set) => ({
  datasets: [],
  models: [],
  pipelines: [],
  selectedDataset: null,
  selectedModel: null,
  loading: false,
  error: null,
  currentStep: 'data',
  pipelineConfig: {},

  setDatasets: (datasets) => set({ datasets }),
  addDataset: (dataset) =>
    set((state) => ({ datasets: [...state.datasets, dataset] })),
  removeDataset: (id) =>
    set((state) => ({ datasets: state.datasets.filter((d) => d.id !== id) })),
  selectDataset: (dataset) => set({ selectedDataset: dataset }),

  setModels: (models) => set({ models }),
  addModel: (model) =>
    set((state) => ({ models: [...state.models, model] })),
  removeModel: (id) =>
    set((state) => ({ models: state.models.filter((m) => m.id !== id) })),
  selectModel: (model) => set({ selectedModel: model }),

  setPipelines: (pipelines) => set({ pipelines }),
  addPipeline: (pipeline) =>
    set((state) => ({ pipelines: [...state.pipelines, pipeline] })),
  updatePipeline: (id, pipeline) =>
    set((state) => ({
      pipelines: state.pipelines.map((p) => (p.pipeline_id === id ? pipeline : p)),
    })),

  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),

  setCurrentStep: (currentStep) => set({ currentStep }),

  updatePipelineConfig: (config) =>
    set((state) => ({
      pipelineConfig: { ...state.pipelineConfig, ...config },
    })),
}))

export default usePipelineStore

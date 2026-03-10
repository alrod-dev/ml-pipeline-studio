import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import DataExplorer from './pages/DataExplorer'
import PipelineBuilder from './pages/PipelineBuilder'
import TrainingView from './pages/TrainingView'
import ModelComparison from './pages/ModelComparison'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/data-explorer" element={<DataExplorer />} />
          <Route path="/pipeline-builder" element={<PipelineBuilder />} />
          <Route path="/training/:pipelineId" element={<TrainingView />} />
          <Route path="/models/comparison" element={<ModelComparison />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App

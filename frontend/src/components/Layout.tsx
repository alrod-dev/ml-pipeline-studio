import React from 'react'
import { Link } from 'react-router-dom'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-blue-600 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center space-x-3">
              <div className="text-2xl font-bold">ML Pipeline Studio</div>
            </Link>
            <div className="flex space-x-6">
              <Link to="/" className="hover:text-blue-200 transition">Dashboard</Link>
              <Link to="/data-explorer" className="hover:text-blue-200 transition">Data Explorer</Link>
              <Link to="/pipeline-builder" className="hover:text-blue-200 transition">Pipeline Builder</Link>
              <Link to="/models/comparison" className="hover:text-blue-200 transition">Models</Link>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {children}
      </main>

      <footer className="bg-gray-800 text-gray-300 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-8 text-center">
          <p>ML Pipeline Studio © 2024 | Built by Alfredo Wiesner</p>
        </div>
      </footer>
    </div>
  )
}

export default Layout

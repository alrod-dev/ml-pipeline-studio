# ML Pipeline Studio

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React 18+](https://img.shields.io/badge/react-18+-blue.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

An end-to-end ML Pipeline Builder with a visual interface for building, training, and deploying machine learning models. Built with Python/FastAPI backend and React/TypeScript frontend.

## Features

- **Dataset Management**: Upload, explore, and analyze datasets (CSV, JSON, XLSX)
- **Visual Pipeline Builder**: Drag-and-drop interface for pipeline configuration
- **Preprocessing**: Feature scaling, categorical encoding, imputation, feature selection
- **Multiple Algorithms**: Support for classification, regression, and clustering models
- **Model Training**: Train models with custom hyperparameters
- **Evaluation Metrics**: Comprehensive metrics including accuracy, F1, precision, recall, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves, feature importance, learning curves
- **Model Export**: Export trained models in JOBLIB or ONNX formats
- **REST API**: Complete API for programmatic access

## Architecture

```mermaid
graph LR
    A[Dataset Upload] --> B[Data Exploration]
    B --> C[Pipeline Configuration]
    C --> D[Preprocessing]
    D --> E[Model Training]
    E --> F[Evaluation]
    F --> G[Visualization]
    G --> H[Model Export]
```

## Tech Stack

### Backend
- **Framework**: FastAPI 0.104+
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Model Export**: joblib, ONNX, skl2onnx
- **Server**: Uvicorn
- **Database**: In-memory (production would use PostgreSQL)

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Charts**: Recharts
- **HTTP Client**: Axios
- **Routing**: React Router v6

## Project Structure

```
ml-pipeline-studio/
├── backend/
│   ├── models/
│   │   └── schemas.py           # Pydantic models
│   ├── routers/
│   │   ├── datasets.py          # Dataset endpoints
│   │   ├── pipelines.py         # Pipeline endpoints
│   │   └── models.py            # Model endpoints
│   ├── services/
│   │   ├── data_loader.py       # Load and validate data
│   │   ├── preprocessor.py      # Feature engineering
│   │   ├── trainer.py           # Model training
│   │   ├── evaluator.py         # Model evaluation
│   │   ├── pipeline_builder.py  # Orchestrate pipelines
│   │   └── exporter.py          # Model export
│   ├── utils/
│   │   └── visualization.py     # Chart generation
│   ├── tests/
│   │   ├── test_trainer.py
│   │   └── test_preprocessor.py
│   ├── main.py                  # FastAPI app
│   ├── config.py                # Configuration
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── DataExplorer.tsx
│   │   │   ├── PipelineBuilder.tsx
│   │   │   ├── TrainingView.tsx
│   │   │   └── ModelComparison.tsx
│   │   ├── components/
│   │   │   ├── Layout.tsx
│   │   │   ├── DatasetStats.tsx
│   │   │   └── ... (visualization components)
│   │   ├── store/
│   │   │   └── pipelineStore.ts
│   │   ├── lib/
│   │   │   └── api.ts
│   │   ├── types/
│   │   │   └── index.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
├── sample-data/
│   └── iris.csv
├── docker-compose.yml
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose (optional)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/alrod-dev/ml-pipeline-studio.git
cd ml-pipeline-studio
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Backend will run on `http://localhost:8000`

3. **Frontend Setup** (in another terminal)
```bash
cd frontend
npm install
npm run dev
```

Frontend will run on `http://localhost:3000`

### Docker Setup

```bash
docker-compose up
```

This will start:
- Backend API on `http://localhost:8000`
- Frontend on `http://localhost:3000`

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Key Endpoints

#### Datasets
- `POST /api/datasets/upload` - Upload dataset
- `GET /api/datasets` - List datasets
- `GET /api/datasets/{id}` - Get dataset metadata
- `GET /api/datasets/{id}/stats` - Get statistics
- `GET /api/datasets/{id}/preview` - Preview data

#### Pipelines
- `POST /api/pipelines/run` - Run pipeline
- `GET /api/pipelines` - List pipelines
- `GET /api/pipelines/{id}` - Get pipeline results
- `GET /api/pipelines/{id}/status` - Get status

#### Models
- `GET /api/models` - List models
- `GET /api/models/{id}` - Get model info
- `POST /api/models/{id}/export` - Export model
- `POST /api/models/{id}/predict` - Make predictions

## Supported Models

### Classification
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Logistic Regression

### Regression
- Linear Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

### Clustering
- K-Means

## Preprocessing Options

- **Scaling**: Standard, MinMax, Robust
- **Encoding**: Label, One-Hot
- **Imputation**: Mean, Median, Most Frequent, Drop
- **Feature Selection**: SelectKBest, RFE

## Example Usage

### Python Client

```python
import requests

# Upload dataset
files = {'file': open('data.csv', 'rb')}
resp = requests.post('http://localhost:8000/api/datasets/upload', files=files)
dataset_id = resp.json()['id']

# Run pipeline
config = {
    'dataset_id': dataset_id,
    'target_column': 'target',
    'model_type': 'random_forest',
    'problem_type': 'classification',
    'preprocessing': {
        'scaling': {'type': 'standard'},
        'encoding': {'type': 'label'}
    }
}
resp = requests.post('http://localhost:8000/api/pipelines/run', json=config)
result = resp.json()

# Get results
pipeline_id = result['pipeline_id']
resp = requests.get(f'http://localhost:8000/api/pipelines/{pipeline_id}')
print(resp.json())
```

## Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=services
```

### Frontend Tests
```bash
cd frontend
npm run test
```

## Project Statistics

- **Backend**: 15+ modules with 2000+ lines of production code
- **Frontend**: 10+ React components with TypeScript
- **Tests**: Unit tests for core services
- **Models**: 6+ ML algorithms across 3 problem types
- **Preprocessing**: 4 major preprocessing categories

## Performance Considerations

- Datasets up to 100 MB supported
- Optimized batch processing for large datasets
- Vectorized operations using numpy/pandas
- Asynchronous API responses
- Caching for repeated operations

## Security Features

- CORS enabled for specified origins
- Input validation on all endpoints
- Error handling with safe error messages
- File upload validation
- Configuration via environment variables

## Future Enhancements

- Database integration (PostgreSQL)
- Advanced hyperparameter tuning
- Cross-validation support
- Model interpretability (SHAP, LIME)
- Automated ML (AutoML)
- Model versioning and tracking
- Distributed training support
- Real-time training progress websockets
- Model serving endpoints

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Alfredo Wiesner** ([@alrod-dev](https://github.com/alrod-dev))
- Senior Software Engineer with 8+ years experience
- ML/Data Science focus
- Full-stack development expertise

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- UI powered by [React](https://react.dev/) and [Tailwind CSS](https://tailwindcss.com/)
- ML models from [scikit-learn](https://scikit-learn.org/)
- Visualizations with [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/)

## Contact

For questions or support, please open an issue on GitHub.

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function fetchApi(path: string, options?: RequestInit) {
  try {
    const res = await fetch(`${API_BASE}${path}`, options);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  } catch {
    return [];
  }
}

export const datasets = {
  list: () => fetchApi('/api/datasets'),
  get: (id: string) => fetchApi(`/api/datasets/${id}`),
  create: (data: any) => fetchApi('/api/datasets', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  delete: (id: string) => fetchApi(`/api/datasets/${id}`, { method: 'DELETE' }),
  getStats: (id: string) => fetchApi(`/api/datasets/${id}/stats`),
};

export const pipelines = {
  list: () => fetchApi('/api/pipelines'),
  get: (id: string) => fetchApi(`/api/pipelines/${id}`),
  create: (data: any) => fetchApi('/api/pipelines', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  update: (id: string, data: any) => fetchApi(`/api/pipelines/${id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }),
  delete: (id: string) => fetchApi(`/api/pipelines/${id}`, { method: 'DELETE' }),
  run: (id: string) => fetchApi(`/api/pipelines/${id}/run`, { method: 'POST' }),
};

export const models = {
  list: () => fetchApi('/api/models'),
  get: (id: string) => fetchApi(`/api/models/${id}`),
  compare: (ids: string[]) => fetchApi('/api/models/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids }) }),
  getMetrics: (id: string) => fetchApi(`/api/models/${id}/metrics`),
  deploy: (id: string) => fetchApi(`/api/models/${id}/deploy`, { method: 'POST' }),
};

export const training = {
  list: () => fetchApi('/api/training'),
  get: (id: string) => fetchApi(`/api/training/${id}`),
  start: (config: any) => fetchApi('/api/training', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) }),
  stop: (id: string) => fetchApi(`/api/training/${id}/stop`, { method: 'POST' }),
  getLogs: (id: string) => fetchApi(`/api/training/${id}/logs`),
};

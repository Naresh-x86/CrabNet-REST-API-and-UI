'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getPSMILESModels, getPSMILESProperties, validatePSMILES } from '@/lib/api';
import type { PSMILESModelInfo, PSMILESPropertyInfo } from '@/lib/types';
import { ErrorMessage, LoadingSpinner } from '@/components';

// Custom display names for models (frontend aliases)
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'v1': 'BioCrabNet Transformer-Based Model',
  'v2': 'Fingerprint Model',
  'v3': 'Ensemble Model',
};

// Helper to get display name for a model
function getModelDisplayName(modelId: string, fallback: string = ''): string {
  return MODEL_DISPLAY_NAMES[modelId] || fallback;
}

export default function PolymerPage() {
  const router = useRouter();

  // State
  const [models, setModels] = useState<PSMILESModelInfo[]>([]);
  const [properties, setProperties] = useState<PSMILESPropertyInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<PSMILESModelInfo | null>(null);
  const [smilesInput, setSmilesInput] = useState<string>('');
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Example SMILES for quick testing
  const exampleSmiles = [
    { name: 'Polystyrene', smiles: '*CC(*)c1ccccc1' },
    { name: 'PMMA', smiles: '*CC(*)(C)C(=O)OC' },
    { name: 'Polyethylene', smiles: '*CC*' },
    { name: 'Polypropylene', smiles: '*CC(*)C' },
    { name: 'PVC', smiles: '*CC(*)Cl' },
    { name: 'Polyacrylonitrile', smiles: '*CC(*)C#N' },
  ];

  // Load models and properties on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [modelList, propList] = await Promise.all([
          getPSMILESModels(),
          getPSMILESProperties()
        ]);
        setModels(modelList); // Show all models including V1
        setProperties(propList);
        setError(null);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : 'Failed to load models. Is the API server running?'
        );
      } finally {
        setIsLoadingModels(false);
      }
    };

    loadData();
  }, []);

  // Handle prediction submission
  const handlePredict = async () => {
    if (!smilesInput.trim()) {
      setError('Please enter a SMILES string');
      return;
    }

    if (!selectedModel) {
      setError('Please select a model first');
      return;
    }

    setIsValidating(true);
    setError(null);

    try {
      // Validate SMILES first
      const validation = await validatePSMILES(smilesInput.trim());
      
      if (!validation.valid) {
        setError(validation.error || 'Invalid SMILES structure');
        return;
      }

      // Navigate to results page
      const params = new URLSearchParams({
        smiles: smilesInput.trim(),
        modelVersion: selectedModel.id,
        modelName: getModelDisplayName(selectedModel.id, selectedModel.name),
        modelDescription: selectedModel.description,
        propertyId: 'glass_transition_temperature',
        propertyName: properties[0]?.name || 'Glass Transition Temperature',
        propertyUnits: properties[0]?.units || '°C',
      });

      router.push(`/polymer/results?${params.toString()}`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Validation failed. Please try again.'
      );
    } finally {
      setIsValidating(false);
    }
  };

  // Handle example click
  const handleExampleClick = (smiles: string) => {
    setSmilesInput(smiles);
  };

  return (
    <div className="flex min-h-screen flex-col bg-gradient-to-b from-zinc-50 to-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white/80 backdrop-blur-md">
        <div className="mx-auto max-w-6xl px-6 py-6">
          <div className="flex items-center gap-4">
            {/* Logo Icon */}
            <img
              src="https://crabnet.readthedocs.io/en/latest/_static/logo.png"
              alt="CrabNet logo"
              className="h-12 w-12 object-contain"
            />
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-zinc-900">
                BioCrabNet Polymer Property Predictor
              </h1>
              <p className="text-sm text-zinc-500">
                CrabNet adapted for polymer SMILES prediction
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-10">
        {/* Error Display */}
        {error && (
          <div className="mb-8">
            <ErrorMessage
              message={error}
              onRetry={() => {
                setError(null);
                if (isLoadingModels) {
                  window.location.reload();
                }
              }}
            />
          </div>
        )}

        {/* Instructions Card */}
        <div className="mb-10 rounded-2xl border border-zinc-200 bg-white p-8 shadow-sm">
          <h2 className="mb-6 text-xl font-semibold text-zinc-900">
            How to Use
          </h2>
          <div className="grid gap-6 md:grid-cols-3">
            <div className="flex gap-4">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-gray-100 text-lg font-bold text-orange-800">
                1
              </div>
              <div>
                <h3 className="font-medium text-zinc-900">
                  Select Model
                </h3>
                <p className="mt-1 text-sm text-zinc-500">
                  Choose which BioCrabNet model version to use for prediction
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-gray-100 text-lg font-bold text-orange-800">
                2
              </div>
              <div>
                <h3 className="font-medium text-zinc-900">
                  Enter SMILES
                </h3>
                <p className="mt-1 text-sm text-zinc-500">
                  Input the polymer repeat unit SMILES (use * for connection points)
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-gray-100 text-lg font-bold text-orange-800">
                3
              </div>
              <div>
                <h3 className="font-medium text-zinc-900">
                  View Results
                </h3>
                <p className="mt-1 text-sm text-zinc-500">
                  Get predicted Tg, structure visualization, and molecular descriptors
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Model Selector */}
        <div className="mb-8">
          <label className="mb-3 block text-sm font-medium text-zinc-700">
            Step 1: Select a model version
          </label>
          <ModelSelector
            models={models}
            selectedModel={selectedModel}
            onSelect={setSelectedModel}
            isLoading={isLoadingModels}
          />
        </div>

        {/* SMILES Input */}
        <div className="mb-10">
          <label className="mb-3 block text-sm font-medium text-zinc-700">
            Step 2: Enter polymer SMILES
          </label>
          <SMILESInput
            value={smilesInput}
            onChange={setSmilesInput}
            onSubmit={handlePredict}
            isValidating={isValidating}
            examples={exampleSmiles}
            onExampleClick={handleExampleClick}
          />
        </div>

        {!selectedModel && smilesInput && (
          <div className="mt-4 rounded-lg border border-sky-200 bg-sky-50 p-4">
            <div className="flex items-center gap-2">
              <svg
                className="h-5 w-5 text-sky-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              <span className="font-medium text-sky-800">
                Please select a model first
              </span>
            </div>
            <p className="mt-1 text-sm text-sky-700">
              Choose a model version before predicting
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-200 bg-white/50 py-6">
        <div className="mx-auto max-w-6xl px-6 text-center text-sm text-zinc-500">
          <p>
            BioCrabNet • CrabNet adapted for polymer property prediction
          </p>
        </div>
      </footer>
    </div>
  );
}

// Model Selector Component (similar to PropertySelector)
function ModelSelector({
  models,
  selectedModel,
  onSelect,
  isLoading,
}: {
  models: PSMILESModelInfo[];
  selectedModel: PSMILESModelInfo | null;
  onSelect: (model: PSMILESModelInfo) => void;
  isLoading: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);

  if (isLoading) {
    return (
      <div className="flex h-16 items-center justify-center rounded-xl border border-zinc-200 bg-white">
        <LoadingSpinner size="sm" message="Loading models..." />
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between rounded-xl border border-zinc-200 bg-white px-5 py-4 text-left shadow-sm transition-all hover:border-zinc-300 hover:shadow-md"
      >
        <div>
          <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
            Model Version
          </span>
          <p className="mt-1 text-lg font-medium text-zinc-900">
            {selectedModel ? getModelDisplayName(selectedModel.id, selectedModel.name) : 'Click to select a model'}
          </p>
        </div>
        <svg
          className={`h-5 w-5 text-zinc-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute z-10 mt-2 w-full rounded-xl border border-zinc-200 bg-white py-2 shadow-lg">
          {models.map((model) => (
            <button
              key={model.id}
              onClick={() => {
                onSelect(model);
                setIsOpen(false);
              }}
              className={`flex w-full items-center justify-between px-5 py-3 text-left transition-colors hover:bg-zinc-50 ${
                selectedModel?.id === model.id ? 'bg-orange-50' : ''
              }`}
            >
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <p className="font-medium text-zinc-900">{getModelDisplayName(model.id, model.name)}</p>
                </div>
                <p className="text-sm text-zinc-500">{model.description}</p>
              </div>
              {selectedModel?.id === model.id && (
                <svg className="h-5 w-5 text-orange-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// SMILES Input Component (similar to SearchBar)
function SMILESInput({
  value,
  onChange,
  onSubmit,
  isValidating,
  examples,
  onExampleClick,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isValidating: boolean;
  examples: { name: string; smiles: string }[];
  onExampleClick: (smiles: string) => void;
}) {
  return (
    <div>
      <div className="flex gap-4">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') onSubmit();
          }}
          placeholder="Enter polymer SMILES (e.g., *CC(*)c1ccccc1)"
          className="flex-1 rounded-xl border border-zinc-200 bg-white px-5 py-4 text-lg shadow-sm transition-all placeholder:text-zinc-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
        />
        <button
          onClick={onSubmit}
          disabled={isValidating || !value.trim()}
          className="flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-4 font-semibold text-white shadow-sm transition-all hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isValidating ? (
            <>
              <LoadingSpinner size="sm" />
              Validating...
            </>
          ) : (
            <>
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Predict
            </>
          )}
        </button>
      </div>

      {/* Example polymers */}
      <div className="mt-4">
        <p className="mb-2 text-sm text-zinc-500">Examples:</p>
        <div className="flex flex-wrap gap-2">
          {examples.map((example) => (
            <button
              key={example.name}
              onClick={() => onExampleClick(example.smiles)}
              className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-1.5 text-sm transition-all hover:border-zinc-300 hover:bg-zinc-100"
            >
              <span className="font-medium text-zinc-700">{example.name}</span>
              <span className="ml-1.5 font-mono text-zinc-500">{example.smiles}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

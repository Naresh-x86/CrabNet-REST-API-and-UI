'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getModels, retrieveMaterials } from '@/lib/api';
import type { ModelInfo, MaterialBasicInfo } from '@/lib/types';
import {
  PropertySelector,
  SearchBar,
  MaterialsTable,
  ErrorMessage,
} from '@/components';

export default function Home() {
  const router = useRouter();

  // State
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [searchResults, setSearchResults] = useState<MaterialBasicInfo[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  // Load models on mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const modelList = await getModels();
        setModels(modelList);
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

    loadModels();
  }, []);

  // Handle search
  const handleSearch = async (formula: string) => {
    setIsSearching(true);
    setError(null);
    setHasSearched(true);

    try {
      const response = await retrieveMaterials(formula);
      setSearchResults(response.data || []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Search failed. Please try again.'
      );
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  // Handle material selection - navigate to detail page
  const handleSelectMaterial = (material: MaterialBasicInfo) => {
    if (!selectedModel) {
      setError('Please select a property to predict first.');
      return;
    }

    // Navigate to results page with query params
    const params = new URLSearchParams({
      materialId: material.material_id,
      formula: material.formula_pretty,
      propertyName: selectedModel.name,
      propertyDescription: selectedModel.description,
      propertyUnits: selectedModel.units,
    });

    router.push(`/results?${params.toString()}`);
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
                CrabNet Material Property Predictor
              </h1>
              <p className="text-sm text-zinc-500">
                Compositionally-restricted attention-based Network
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
                  Select Property
                </h3>
                <p className="mt-1 text-sm text-zinc-500">
                  Choose which material property you want to predict from
                  available options
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-gray-100 text-lg font-bold text-orange-800">
                2
              </div>
              <div>
                <h3 className="font-medium text-zinc-900">
                  Search Material
                </h3>
                <p className="mt-1 text-sm text-zinc-500">
                  Enter a chemical formula to find matching materials in the
                  database
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
                  Click on a material to see detailed predictions and properties
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Property Selector */}
        <div className="mb-8">
          <label className="mb-3 block text-sm font-medium text-zinc-700">
            Step 1: Select a property to predict
          </label>
          <PropertySelector
            models={models}
            selectedModel={selectedModel}
            onSelect={setSelectedModel}
            isLoading={isLoadingModels}
          />
        </div>

        {/* Search Bar */}
        <div className="mb-10">
          <label className="mb-3 block text-sm font-medium text-zinc-700">
            Step 2: Search for a material by chemical formula
          </label>
          <SearchBar onSearch={handleSearch} isSearching={isSearching} />
        </div>

        {/* Search Results */}
        {hasSearched && (
          <div>
            {searchResults.length === 0 && !isSearching && !error ? (
              <div className="rounded-xl border border-zinc-200 bg-white p-12 text-center">
                <svg
                  className="mx-auto h-12 w-12 text-zinc-300"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <h3 className="mt-4 text-lg font-medium text-zinc-900">
                  No materials found
                </h3>
                <p className="mt-2 text-zinc-500">
                  Try a different chemical formula or check your spelling
                </p>
              </div>
            ) : (
              <MaterialsTable
                materials={searchResults}
                onSelectMaterial={handleSelectMaterial}
                isLoading={isSearching}
              />
            )}

            {!selectedModel && searchResults.length > 0 && (
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
                    Please select a property first
                  </span>
                </div>
                <p className="mt-1 text-sm text-sky-700">
                  Choose a property to predict before selecting a material
                </p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-200 bg-white/50 py-6">
        <div className="mx-auto max-w-6xl px-6 text-center text-sm text-zinc-500">
          <p>
            CrabNet • Compositionally-restricted attention-based Network
          </p>
        </div>
      </footer>
    </div>
  );
}

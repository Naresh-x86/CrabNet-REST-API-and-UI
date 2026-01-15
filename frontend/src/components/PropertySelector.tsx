'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import type { ModelInfo } from '@/lib/types';
import { groupModelsByCategory } from '@/lib/utils';

interface PropertySelectorProps {
  models: ModelInfo[];
  selectedModel: ModelInfo | null;
  onSelect: (model: ModelInfo) => void;
  isLoading?: boolean;
}

export function PropertySelector({
  models,
  selectedModel,
  onSelect,
  isLoading = false,
}: PropertySelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Group and filter models
  const groupedModels = useMemo(() => {
    const filtered = models.filter(
      (model) =>
        model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.description.toLowerCase().includes(searchQuery.toLowerCase())
    );
    return groupModelsByCategory(filtered);
  }, [models, searchQuery]);

  // Close modal on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false);
    };
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const handleSelect = useCallback(
    (model: ModelInfo) => {
      onSelect(model);
      setIsOpen(false);
      setSearchQuery('');
    },
    [onSelect]
  );

  return (
    <>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(true)}
        disabled={isLoading}
        className="group flex w-full items-center justify-between gap-3 rounded-xl border-2 border-zinc-200 bg-white px-5 py-4 text-left transition-all hover:border-gray-300 hover:shadow-lg focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:cursor-not-allowed disabled:opacity-50:border-blue-500"
      >
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
            Property to Predict
          </span>
          {selectedModel ? (
            <div>
              <span className="text-lg font-semibold text-zinc-900">
                {selectedModel.name}
              </span>
              <span className="ml-2 text-sm text-zinc-500">
                ({selectedModel.units})
              </span>
            </div>
          ) : (
            <span className="text-lg text-zinc-400">
              {isLoading ? 'Loading models...' : 'Click to select a property'}
            </span>
          )}
          {selectedModel && (
            <span className="text-sm text-zinc-500">
              {selectedModel.description}
            </span>
          )}
        </div>
        <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-gray-100 text-orange-800 transition-colors group-hover:bg-orange-200/60">
          <svg
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </button>

      {/* Modal */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          />

          {/* Modal Content */}
          <div className="relative max-h-[85vh] w-full max-w-4xl overflow-hidden rounded-2xl bg-white shadow-2xl">
            {/* Header */}
            <div className="sticky top-0 z-10 border-b border-zinc-200 bg-white/95 px-6 py-4 backdrop-blur-sm">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-zinc-900">
                  Select Property to Predict
                </h2>
                <button
                  onClick={() => setIsOpen(false)}
                  className="rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700:bg-zinc-800:text-zinc-300"
                >
                  <svg
                    className="h-5 w-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>

              {/* Search */}
              <div className="relative mt-4">
                <input
                  type="text"
                  placeholder="Search properties..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full rounded-lg border border-zinc-200 bg-zinc-50 px-4 py-3 pl-10 text-zinc-900 placeholder-zinc-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                  autoFocus
                />
                <svg
                  className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-zinc-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  />
                </svg>
              </div>

              <p className="mt-2 text-sm text-zinc-500">
                {models.length} properties available • Select one to predict
              </p>
            </div>

            {/* Categories */}
            <div className="max-h-[60vh] overflow-y-auto px-6 py-4">
              {Object.entries(groupedModels).map(([category, categoryModels]) => (
                <div key={category} className="mb-6">
                  <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-500">
                    <span className="h-px flex-1 bg-zinc-200" />
                    {category}
                    <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs">
                      {categoryModels.length}
                    </span>
                    <span className="h-px flex-1 bg-zinc-200" />
                  </h3>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {categoryModels.map((model) => (
                      <button
                        key={model.name}
                        onClick={() => handleSelect(model)}
                        className={`group flex flex-col rounded-lg border-2 p-3 text-left transition-all hover:border-blue-400 hover:bg-blue-50 hover:shadow-md:bg-blue-900/20 ${
                          selectedModel?.name === model.name
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-zinc-200 bg-white'
                        }`}
                      >
                        <span className="font-medium text-zinc-900">
                          {model.name}
                        </span>
                        <span className="mt-1 text-xs text-zinc-500">
                          {model.description}
                        </span>
                        <span className="mt-2 inline-flex self-start rounded-full bg-zinc-100 px-2 py-0.5 text-xs font-medium text-zinc-600">
                          {model.units}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              ))}

              {Object.keys(groupedModels).length === 0 && (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <svg
                    className="h-12 w-12 text-zinc-300"
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
                  <p className="mt-4 text-zinc-500">
                    No properties found matching &ldquo;{searchQuery}&rdquo;
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

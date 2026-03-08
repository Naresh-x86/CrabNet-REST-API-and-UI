'use client';

import { useSearchParams, useRouter } from 'next/navigation';
import { useState, useEffect, Suspense } from 'react';
import { predictPSMILES } from '@/lib/api';
import type { PSMILESPredictionResult, PSMILESDescriptor } from '@/lib/types';
import { BackButton, LoadingSpinner, ErrorMessage } from '@/components';

function PolymerResultsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  // Extract query params
  const smiles = searchParams.get('smiles') || '';
  const modelVersion = searchParams.get('modelVersion') || 'v3';
  const modelName = searchParams.get('modelName') || 'FingerprintNet Ensemble (V3)';
  const propertyId = searchParams.get('propertyId') || 'glass_transition_temperature';
  const propertyName = searchParams.get('propertyName') || 'Glass Transition Temperature';
  const propertyUnits = searchParams.get('propertyUnits') || '°C';

  // State
  const [prediction, setPrediction] = useState<PSMILESPredictionResult | null>(null);
  const [structureImage, setStructureImage] = useState<string | null>(null);
  const [descriptors, setDescriptors] = useState<PSMILESDescriptor[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!smiles) {
      setError('No SMILES provided');
      setIsLoading(false);
      return;
    }

    const loadResults = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // The prediction endpoint returns everything we need
        const predResult = await predictPSMILES(smiles, modelVersion, propertyId);

        setPrediction(predResult);
        setStructureImage(predResult.structure_image);
        setDescriptors(predResult.descriptors);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to load prediction results'
        );
      } finally {
        setIsLoading(false);
      }
    };

    loadResults();
  }, [smiles, modelVersion, propertyId]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <LoadingSpinner size="lg" message="Loading prediction results..." />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-gradient-to-b from-zinc-50 to-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white/80 backdrop-blur-md">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <img
                src="https://crabnet.readthedocs.io/en/latest/_static/logo.png"
                alt="CrabNet Logo"
                className="h-10 w-10 object-contain"
              />
              <span className="text-lg font-bold text-zinc-900">
                BioCrabNet Polymer Property Predictor
              </span>
            </div>
            <BackButton onClick={() => router.push('/polymer')} label="New Search" />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-8">
        {error && (
          <div className="mb-8">
            <ErrorMessage message={error} onRetry={() => window.location.reload()} />
          </div>
        )}

        {/* Top Section: Prediction + Structure Viewer */}
        <div className="mb-8 grid gap-8 lg:grid-cols-2">
          {/* Left: Prediction Highlights */}
          <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-lg">
            <div className="mb-6 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100">
                <svg
                  className="h-5 w-5 text-orange-800"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <h2 className="text-xl font-bold text-zinc-900">
                Prediction Results
              </h2>
            </div>

            {/* SMILES */}
            <div className="mb-6">
              <span className="text-sm font-medium uppercase tracking-wider text-zinc-500">
                Polymer SMILES
              </span>
              <div className="mt-1">
                <code className="text-2xl font-bold text-zinc-900 font-mono">
                  {smiles}
                </code>
              </div>
            </div>

            {/* Property Being Predicted */}
            <div className="mb-6">
              <span className="text-sm font-medium uppercase tracking-wider text-zinc-500">
                Predicted Property
              </span>
              <p className="mt-1 text-xl font-semibold text-zinc-800">
                {propertyName}
              </p>
              <p className="text-sm text-zinc-500">
                Model: {modelName}
              </p>
            </div>

            {/* Predicted Value */}
            <div className="mb-6 rounded-xl bg-gradient-to-r from-orange-800 to-orange-800 p-6 text-white">
              <span className="text-sm font-medium uppercase tracking-wider text-blue-100">
                Predicted Value
              </span>
              {prediction ? (
                <div className="mt-2">
                  <span className="text-5xl font-bold">
                    {prediction.predicted_value.toFixed(2)}
                  </span>
                  <span className="ml-2 text-2xl font-medium text-blue-100">
                    {propertyUnits}
                  </span>
                </div>
              ) : (
                <p className="mt-2 text-blue-100">Prediction unavailable</p>
              )}
            </div>

            {/* Uncertainty */}
            <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-4">
              <span className="text-sm font-medium uppercase tracking-wider text-zinc-500">
                Uncertainty (±)
              </span>
              {prediction && prediction.uncertainty > 0 ? (
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-2xl font-bold text-zinc-900">
                    ± {prediction.uncertainty.toFixed(2)}
                  </span>
                  <span className="text-lg text-zinc-500">
                    {propertyUnits}
                  </span>
                </div>
              ) : (
                <p className="mt-1 text-zinc-500">N/A</p>
              )}
            </div>
          </div>

          {/* Right: 2D Structure Viewer */}
          <div className="flex flex-col rounded-2xl border border-zinc-200 bg-white p-6 shadow-lg">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100">
                <svg
                  className="h-5 w-5 text-orange-800"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5"
                  />
                </svg>
              </div>
              <h2 className="text-xl font-bold text-zinc-900">
                Repeat Unit Structure
              </h2>
            </div>
            <div className="flex flex-1 items-center justify-center overflow-hidden rounded-xl bg-white" style={{ minHeight: '400px' }}>
              {structureImage ? (
                <img
                  src={structureImage}
                  alt="Polymer structure"
                  className="max-h-96 max-w-full object-contain"
                />
              ) : (
                <div className="flex flex-col items-center justify-center text-zinc-400">
                  <svg className="h-16 w-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="mt-2">Structure image unavailable</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Molecular Descriptors Table */}
        <div className="mb-8 rounded-2xl border border-zinc-200 bg-white p-8 shadow-lg">
          <div className="mb-6 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100">
              <svg
                className="h-5 w-5 text-orange-800"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"
                />
              </svg>
            </div>
            <h2 className="text-xl font-bold text-zinc-900">
              Molecular Descriptors
            </h2>
          </div>

          {descriptors.length > 0 ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {descriptors.map((desc) => (
                <DescriptorRow
                  key={desc.name}
                  label={desc.description}
                  value={formatDescriptorValue(desc.value, desc.units)}
                  icon={getDescriptorIcon(desc.name)}
                />
              ))}
            </div>
          ) : (
            <p className="text-zinc-500">
              No descriptor data available
            </p>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-200 bg-white/50 py-6">
        <div className="mx-auto max-w-7xl px-6 text-center text-sm text-zinc-500">
          <p>
            BioCrabNet • CrabNet adapted for polymer property prediction
          </p>
        </div>
      </footer>
    </div>
  );
}

// Descriptor Row Component (matches PropertyRow from results page)
function DescriptorRow({
  label,
  value,
  icon,
  fullWidth = false,
}: {
  label: string;
  value: string;
  icon: string;
  fullWidth?: boolean;
}) {
  return (
    <div
      className={`flex items-center gap-3 rounded-lg border border-zinc-200 bg-zinc-50 p-4 ${fullWidth ? 'sm:col-span-2 lg:col-span-3' : ''}`}
    >
      <span className="text-2xl">{icon}</span>
      <div className="flex-1 min-w-0">
        <span className="block text-xs font-medium uppercase tracking-wider text-zinc-500">
          {label}
        </span>
        <span className="block truncate font-semibold text-zinc-900">
          {value}
        </span>
      </div>
    </div>
  );
}

// Helper to format descriptor values
function formatDescriptorValue(value: number | null, units: string): string {
  if (value === null) return 'N/A';
  if (Number.isInteger(value)) {
    return units === 'count' ? value.toString() : `${value} ${units}`;
  }
  return units === 'ratio' ? value.toFixed(3) : `${value.toFixed(2)} ${units}`;
}

// Helper to get descriptor icons
function getDescriptorIcon(name: string): string {
  const icons: Record<string, string> = {
    MolWt: '⚖️',
    NumRotatableBonds: '🔄',
    NumAromaticRings: '💎',
    TPSA: '📐',
    FractionCSP3: '🔬',
    NumHBondDonors: '🔵',
    NumHBondAcceptors: '🔴',
    RingCount: '⭕',
    HeavyAtomCount: '⚛️',
  };
  return icons[name] || '📊';
}

export default function PolymerResultsPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center bg-gradient-to-b from-zinc-50 to-zinc-100">
          <LoadingSpinner size="lg" message="Loading..." />
        </div>
      }
    >
      <PolymerResultsContent />
    </Suspense>
  );
}

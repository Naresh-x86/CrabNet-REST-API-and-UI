'use client';

import { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  getMaterialSummary,
  predictProperty,
  getRelatedMaterials,
  getNaturalLanguageSummary,
} from '@/lib/api';
import type {
  MaterialSummary,
  PredictionResult,
  SimilarMaterial,
  NaturalLanguageSummary,
} from '@/lib/types';
import { ChemicalFormula, BackButton, LoadingSpinner, ErrorMessage, CrystalViewer } from '@/components';
import { formatNumber, formatMagneticOrdering } from '@/lib/utils';

function ResultsContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Get params from URL
  const materialId = searchParams.get('materialId') || '';
  const formula = searchParams.get('formula') || '';
  const propertyName = searchParams.get('propertyName') || '';
  const propertyDescription = searchParams.get('propertyDescription') || '';
  const propertyUnits = searchParams.get('propertyUnits') || '';

  // State
  const [materialSummary, setMaterialSummary] = useState<MaterialSummary | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [relatedMaterials, setRelatedMaterials] = useState<SimilarMaterial[]>([]);
  const [nlSummary, setNlSummary] = useState<NaturalLanguageSummary | null>(null);

  const [isLoadingSummary, setIsLoadingSummary] = useState(true);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(true);
  const [isLoadingRelated, setIsLoadingRelated] = useState(true);
  const [isLoadingNlSummary, setIsLoadingNlSummary] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [nlError, setNlError] = useState<string | null>(null);

  // Load data on mount
  useEffect(() => {
    if (!materialId || !formula || !propertyName) {
      setError('Missing required parameters');
      setIsLoadingSummary(false);
      setIsLoadingPrediction(false);
      setIsLoadingRelated(false);
      return;
    }

    // Load material summary
    const loadSummary = async () => {
      try {
        const response = await getMaterialSummary(materialId);
        if (response.data && response.data.length > 0) {
          setMaterialSummary(response.data[0]);
        }
      } catch (err) {
        console.error('Failed to load material summary:', err);
      } finally {
        setIsLoadingSummary(false);
      }
    };

    // Load prediction
    const loadPrediction = async () => {
      try {
        const result = await predictProperty(formula, propertyName);
        setPrediction(result);
      } catch (err) {
        console.error('Failed to load prediction:', err);
        setError(err instanceof Error ? err.message : 'Prediction failed');
      } finally {
        setIsLoadingPrediction(false);
      }
    };

    // Load related materials
    const loadRelated = async () => {
      try {
        const materials = await getRelatedMaterials(materialId);
        setRelatedMaterials(materials);
      } catch (err) {
        console.error('Failed to load related materials:', err);
      } finally {
        setIsLoadingRelated(false);
      }
    };

    loadSummary();
    loadPrediction();
    loadRelated();
  }, [materialId, formula, propertyName]);

  // Handle natural language summary generation
  const handleGenerateNlSummary = async () => {
    setIsLoadingNlSummary(true);
    setNlError(null);
    try {
      const result = await getNaturalLanguageSummary(materialId);
      setNlSummary(result);
    } catch (err) {
      setNlError(
        err instanceof Error ? err.message : 'Failed to generate description'
      );
    } finally {
      setIsLoadingNlSummary(false);
    }
  };

  // Loading state
  if (isLoadingSummary && isLoadingPrediction) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <LoadingSpinner size="lg" message="Loading material data..." />
      </div>
    );
  }

  // Determine dimensionality (simple heuristic)
  const getDimensionality = () => {
    // This is a simplified check - in reality you'd need more sophisticated analysis
    return '3D';
  };

  // Format possible species
  const formatPossibleSpecies = (species: string[] | undefined) => {
    if (!species || species.length === 0) return 'N/A';
    return species.join(', ');
  };

  // Get lattice system from symmetry
  const getLatticeSystem = (crystalSystem: string | undefined) => {
    if (!crystalSystem) return 'N/A';
    // Map crystal systems to lattice systems
    const mapping: Record<string, string> = {
      Cubic: 'Cubic',
      Hexagonal: 'Hexagonal',
      Trigonal: 'Rhombohedral',
      Tetragonal: 'Tetragonal',
      Orthorhombic: 'Orthorhombic',
      Monoclinic: 'Monoclinic',
      Triclinic: 'Triclinic',
    };
    return mapping[crystalSystem] || crystalSystem;
  };

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
                CrabNet Material Property Predictor
              </span>
            </div>
            <BackButton onClick={() => router.push('/')} label="New Search" />
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

        {/* Top Section: Prediction + 3D Viewer */}
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

            {/* Chemical Formula */}
            <div className="mb-6">
              <span className="text-sm font-medium uppercase tracking-wider text-zinc-500">
                Chemical Formula
              </span>
              <div className="mt-1">
                <ChemicalFormula
                  formula={formula}
                  className="text-4xl font-bold text-zinc-900"
                />
                <span className="ml-3 text-lg text-zinc-500">
                  ({materialId})
                </span>
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
                {propertyDescription}
              </p>
            </div>

            {/* Predicted Value */}
            <div className="mb-6 rounded-xl bg-gradient-to-r from-orange-800 to-orange-800 p-6 text-white">
              <span className="text-sm font-medium uppercase tracking-wider text-blue-100">
                Predicted Value
              </span>
              {isLoadingPrediction ? (
                <div className="mt-2">
                  <LoadingSpinner size="sm" />
                </div>
              ) : prediction ? (
                <div className="mt-2">
                  <span className="text-5xl font-bold">
                    {formatNumber(prediction.predicted_value, 4)}
                  </span>
                  <span className="ml-2 text-2xl font-medium text-blue-100">
                    {prediction.units}
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
              {isLoadingPrediction ? (
                <div className="mt-2">
                  <LoadingSpinner size="sm" />
                </div>
              ) : prediction ? (
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-2xl font-bold text-zinc-900">
                    ± {formatNumber(prediction.uncertainty, 4)}
                  </span>
                  <span className="text-lg text-zinc-500">
                    {prediction.units}
                  </span>
                </div>
              ) : (
                <p className="mt-1 text-zinc-500">N/A</p>
              )}
            </div>
          </div>

          {/* Right: 3D Structure Viewer */}
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
                3D Crystal Structure
              </h2>
            </div>
            <div className="flex-1 overflow-hidden rounded-xl" style={{ minHeight: '400px' }}>
              <CrystalViewer
                structure={materialSummary?.structure || null}
                isLoading={isLoadingSummary}
              />
            </div>
          </div>
        </div>

        {/* Observed Properties Table */}
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
              Observed Properties
            </h2>
          </div>

          {isLoadingSummary ? (
            <div className="py-8">
              <LoadingSpinner size="md" message="Loading properties..." />
            </div>
          ) : materialSummary ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <PropertyRow
                label="Band Gap"
                value={`${formatNumber(materialSummary.band_gap, 2)} eV`}
                icon="⚡"
              />
              <PropertyRow
                label="Formation Energy"
                value={`${formatNumber(materialSummary.formation_energy_per_atom, 4)} eV/atom`}
                icon="🔋"
              />
              <PropertyRow
                label="Magnetic Ordering"
                value={formatMagneticOrdering(materialSummary.ordering)}
                icon="🧲"
              />
              <PropertyRow
                label="Total Magnetization"
                value={
                  materialSummary.total_magnetization_normalized_formula_units
                    ? `${formatNumber(materialSummary.total_magnetization_normalized_formula_units, 2)} µB/f.u.`
                    : 'N/A'
                }
                icon="📊"
              />
              <PropertyRow
                label="Experimentally Observed"
                value={materialSummary.theoretical ? 'No' : 'Yes'}
                icon={materialSummary.theoretical ? '🔬' : '✅'}
              />
              <PropertyRow
                label="Volume"
                value={`${formatNumber(materialSummary.volume, 2)} Å³`}
                icon="📦"
              />
              <PropertyRow
                label="Crystal System"
                value={materialSummary.symmetry?.crystal_system || 'N/A'}
                icon="💎"
              />
              <PropertyRow
                label="Lattice System"
                value={getLatticeSystem(materialSummary.symmetry?.crystal_system)}
                icon="🔷"
              />
              <PropertyRow
                label="Number of Atoms"
                value={materialSummary.nsites?.toString() || 'N/A'}
                icon="⚛️"
              />
              <PropertyRow
                label="Density"
                value={`${formatNumber(materialSummary.density, 2)} g·cm⁻³`}
                icon="⚖️"
              />
              <PropertyRow
                label="Dimensionality"
                value={getDimensionality()}
                icon="📐"
              />
              <PropertyRow
                label="Possible Oxidation States"
                value={formatPossibleSpecies(materialSummary.possible_species)}
                icon="🔄"
                fullWidth
              />
            </div>
          ) : (
            <p className="text-zinc-500">
              No property data available
            </p>
          )}
        </div>

        {/* Natural Language Description */}
        <div className="mb-8 rounded-2xl border border-zinc-200 bg-white p-8 shadow-lg">
          <div className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
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
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </div>
              <h2 className="text-xl font-bold text-zinc-900">
                Structure Description
              </h2>
            </div>
            {!nlSummary && !isLoadingNlSummary && (
              <button
                onClick={handleGenerateNlSummary}
                className="inline-flex items-center gap-2 rounded-lg bg-gray-100 px-4 py-2 font-bold text-gray-800 transition-colors hover:bg-indigo-200:bg-indigo-900/50"
              >
                Generate Description
              </button>
            )}
          </div>

          {isLoadingNlSummary ? (
            <div className="py-8">
              <LoadingSpinner
                size="md"
                message="Generating natural language description (this may take a moment)..."
              />
            </div>
          ) : nlSummary ? (
            <div className="rounded-lg bg-zinc-50 p-6">
              <p className="leading-relaxed text-zinc-700">
                {nlSummary.description}
              </p>
            </div>
          ) : nlError ? (
            <ErrorMessage
              message={nlError}
              onRetry={handleGenerateNlSummary}
            />
          ) : (
            <div className="rounded-lg border-2 border-dashed border-zinc-200 p-8 text-center">
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
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <p className="mt-4 text-zinc-500">
                Click the button above to generate a natural language
                description of this crystal structure.
              </p>
              <p className="mt-2 text-sm text-zinc-400">
                This uses Robocrystallographer and may take a few seconds.
              </p>
            </div>
          )}
        </div>

        {/* Related Materials */}
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
                  d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                />
              </svg>
            </div>
            <h2 className="text-xl font-bold text-zinc-900">
              Related Materials
            </h2>
          </div>

          {isLoadingRelated ? (
            <div className="py-8">
              <LoadingSpinner size="md" message="Finding similar materials..." />
            </div>
          ) : relatedMaterials.length > 0 ? (
            <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5">
              {relatedMaterials.map((material) => (
                <RelatedMaterialCard key={material.material_id} material={material} />
              ))}
            </div>
          ) : (
            <p className="text-zinc-500">
              No related materials found
            </p>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-200 bg-white/50 py-6">
        <div className="mx-auto max-w-7xl px-6 text-center text-sm text-zinc-500">
          <p>
            Powered by CrabNet Neural Network • Data from Materials Project
          </p>
        </div>
      </footer>
    </div>
  );
}

// Property Row Component
function PropertyRow({
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

// Related Material Card Component
function RelatedMaterialCard({ material }: { material: SimilarMaterial }) {
  const [imageError, setImageError] = useState(false);

  return (
    <div className="group flex flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white transition-all hover:border-gray-400 hover:shadow-lg:border-blue-500">
      {/* Formula */}
      <div className="border-b border-zinc-200 bg-zinc-50 px-4 py-3">
        <ChemicalFormula
          formula={material.formula}
          className="text-lg font-bold text-zinc-900"
        />
      </div>

      {/* Material ID */}
      <div className="px-4 py-2">
        <span className="font-mono text-sm text-zinc-500">
          {material.material_id}
        </span>
      </div>

      {/* Image */}
      <div className="relative aspect-square bg-zinc-100">
        {!imageError ? (
          <img
            src={material.image_url}
            alt={`Structure of ${material.formula}`}
            className="h-full w-full object-cover"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <svg
              className="h-12 w-12 text-zinc-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5"
              />
            </svg>
          </div>
        )}
      </div>

      {/* Similarity */}
      <div className="px-4 py-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-zinc-500">
            Similarity
          </span>
          <span className="font-bold text-gray-600">
            {material.similarity.toFixed(1)}%
          </span>
        </div>
        <div className="mt-2 h-2 overflow-hidden rounded-full bg-zinc-200">
          <div
            className="h-full rounded-full bg-gradient-to-r from-gray-400 to-gray-400"
            style={{ width: `${material.similarity}%` }}
          />
        </div>
      </div>
    </div>
  );
}

// Main export with Suspense boundary for useSearchParams
export default function ResultsPage() {
  return (
    <Suspense
      fallback={
        <div className="flex min-h-screen items-center justify-center">
          <LoadingSpinner size="lg" message="Loading..." />
        </div>
      }
    >
      <ResultsContent />
    </Suspense>
  );
}

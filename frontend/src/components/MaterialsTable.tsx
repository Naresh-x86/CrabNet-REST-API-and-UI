'use client';

import type { MaterialBasicInfo } from '@/lib/types';
import { ChemicalFormula } from './ChemicalFormula';
import { formatNumber } from '@/lib/utils';

interface MaterialsTableProps {
  materials: MaterialBasicInfo[];
  onSelectMaterial: (material: MaterialBasicInfo) => void;
  isLoading?: boolean;
}

export function MaterialsTable({
  materials,
  onSelectMaterial,
  isLoading = false,
}: MaterialsTableProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="flex flex-col items-center gap-4">
          <svg
            className="h-10 w-10 animate-spin text-blue-500"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <p className="text-zinc-500">
            Searching Materials Project...
          </p>
        </div>
      </div>
    );
  }

  if (materials.length === 0) {
    return null;
  }

  return (
    <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-lg">
      <div className="border-b border-zinc-200 bg-zinc-50 px-6 py-4">
        <h3 className="font-semibold text-zinc-900">
          Search Results
        </h3>
        <p className="text-sm text-zinc-500">
          {materials.length} material{materials.length !== 1 ? 's' : ''} found •
          Click a row to view details and predictions
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-zinc-200 bg-zinc-50/50">
              <th className="whitespace-nowrap px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Formula
              </th>
              <th className="whitespace-nowrap px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500">
                MP-ID
              </th>
              <th className="whitespace-nowrap px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Volume (Å³)
              </th>
              <th className="whitespace-nowrap px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Density (g/cm³)
              </th>
              <th className="whitespace-nowrap px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Crystal System
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-200">
            {materials.map((material) => (
              <tr
                key={material.material_id}
                onClick={() => onSelectMaterial(material)}
                className="cursor-pointer transition-colors hover:bg-blue-50:bg-blue-900/10"
              >
                <td className="whitespace-nowrap px-6 py-4">
                  <ChemicalFormula
                    formula={material.formula_pretty}
                    className="text-lg font-semibold text-zinc-900"
                  />
                </td>
                <td className="whitespace-nowrap px-6 py-4">
                  <span className="inline-flex rounded-lg bg-zinc-100 px-3 py-1 font-mono text-sm font-medium text-zinc-700">
                    {material.material_id}
                  </span>
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-zinc-700">
                  {formatNumber(material.volume, 2)}
                </td>
                <td className="whitespace-nowrap px-6 py-4 text-zinc-700">
                  {formatNumber(material.density, 3)}
                </td>
                <td className="whitespace-nowrap px-6 py-4">
                  <span className="inline-flex rounded-full bg-blue-100 px-3 py-1 text-sm font-medium text-blue-700">
                    {material.symmetry?.crystal_system || 'N/A'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

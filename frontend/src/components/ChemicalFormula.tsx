'use client';

import { parseFormulaForDisplay } from '@/lib/utils';

interface ChemicalFormulaProps {
  formula: string;
  className?: string;
}

/**
 * Component to render chemical formulas with proper subscript formatting.
 * e.g., "Fe2O3" renders as Fe₂O₃
 */
export function ChemicalFormula({ formula, className = '' }: ChemicalFormulaProps) {
  const segments = parseFormulaForDisplay(formula);

  return (
    <span className={className}>
      {segments.map((segment, index) =>
        segment.isSubscript ? (
          <sub key={index} className="text-[0.7em]">
            {segment.text}
          </sub>
        ) : (
          <span key={index}>{segment.text}</span>
        )
      )}
    </span>
  );
}

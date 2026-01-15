'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { autocompleteSearch } from '@/lib/api';
import type { AutocompleteResult } from '@/lib/types';
import { ChemicalFormula } from './ChemicalFormula';

interface SearchBarProps {
  onSearch: (formula: string) => void;
  isSearching?: boolean;
}

export function SearchBar({ onSearch, isSearching = false }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<AutocompleteResult[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced autocomplete - 3.5 seconds to avoid rate limiting
  const fetchSuggestions = useCallback(async (searchTerm: string) => {
    if (searchTerm.length < 1) {
      setSuggestions([]);
      return;
    }

    setIsLoadingSuggestions(true);
    try {
      const results = await autocompleteSearch(searchTerm);
      setSuggestions(results);
    } catch (error) {
      console.error('Autocomplete error:', error);
      setSuggestions([]);
    } finally {
      setIsLoadingSuggestions(false);
    }
  }, []);

  // Handle input change with debounce
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    setSelectedIndex(-1);

    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Set new timer - 3.5 seconds debounce to avoid rate limiting
    if (value.length >= 1) {
      setShowSuggestions(true);
      debounceTimerRef.current = setTimeout(() => {
        fetchSuggestions(value);
      }, 3500);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  };

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || suggestions.length === 0) {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSearch();
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev < suggestions.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev > 0 ? prev - 1 : suggestions.length - 1
        );
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0) {
          selectSuggestion(suggestions[selectedIndex].formula_pretty);
        } else {
          handleSearch();
        }
        break;
      case 'Escape':
        setShowSuggestions(false);
        setSelectedIndex(-1);
        break;
    }
  };

  // Select a suggestion
  const selectSuggestion = (formula: string) => {
    setQuery(formula);
    setShowSuggestions(false);
    setSuggestions([]);
    setSelectedIndex(-1);
    // Trigger search after selection
    onSearch(formula);
  };

  // Handle search
  const handleSearch = () => {
    if (query.trim()) {
      setShowSuggestions(false);
      onSearch(query.trim());
    }
  };

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  return (
    <div className="relative w-full">
      <div className="flex gap-3">
        {/* Search Input */}
        <div className="relative flex-1">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
            placeholder="Enter chemical formula (e.g., Fe2O3, TiO2, LiFePO4)"
            className="w-full rounded-xl border-2 border-zinc-200 bg-white px-5 py-4 pr-12 text-lg text-zinc-900 placeholder-zinc-400 transition-all focus:outline-none focus:ring-0 focus:border-0"
            disabled={isSearching}
          />
          {isLoadingSuggestions && (
            <div className="absolute right-4 top-1/2 -translate-y-1/2">
              <svg
                className="h-5 w-5 animate-spin text-blue-500"
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
            </div>
          )}
        </div>

        {/* Search Button */}
        <button
          onClick={handleSearch}
          disabled={!query.trim() || isSearching}
          className="flex items-center gap-2 rounded-xl bg-blue-500 px-8 py-4 font-semibold text-white transition-all hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50:ring-offset-zinc-900"
        >
          {isSearching ? (
            <>
              <svg
                className="h-5 w-5 animate-spin"
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
              Searching...
            </>
          ) : (
            <>
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
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
              Search
            </>
          )}
        </button>
      </div>

      {/* Autocomplete Suggestions */}
      {showSuggestions && (
        <div
          ref={suggestionsRef}
          className="absolute z-20 mt-2 w-full max-w-full rounded-xl border border-zinc-200 bg-white shadow-xl"
        >
          {suggestions.length > 0 ? (
            <ul className="max-h-64 overflow-y-auto py-2">
              {suggestions.map((suggestion, index) => (
                <li key={suggestion.formula_pretty}>
                  <button
                    onClick={() => selectSuggestion(suggestion.formula_pretty)}
                    className={`flex w-full items-center gap-4 px-4 py-3 text-left transition-colors ${
                      index === selectedIndex
                        ? 'bg-blue-50'
                        : 'hover:bg-zinc-50:bg-zinc-800'
                    }`}
                  >
                    <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-100 text-sm font-medium text-zinc-600">
                      <svg
                        className="h-4 w-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                        />
                      </svg>
                    </span>
                    <ChemicalFormula
                      formula={suggestion.formula_pretty}
                      className="text-lg font-medium text-zinc-900"
                    />
                  </button>
                </li>
              ))}
            </ul>
          ) : isLoadingSuggestions ? (
            <div className="flex items-center justify-center py-8">
              <svg
                className="h-6 w-6 animate-spin text-blue-500"
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
              <span className="ml-3 text-zinc-500">
                Fetching suggestions...
              </span>
            </div>
          ) : (
            <div className="py-6 text-center text-zinc-500">
              <p>Type to search for formulas</p>
              <p className="mt-1 text-xs">
                Suggestions will appear after 3.5 seconds
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

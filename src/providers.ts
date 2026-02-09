// ABOUTME: Embedding provider interface for pluggable embedding backends
// ABOUTME: Supports multiple models with version tracking and dimension detection

import { EmbeddingData } from './embeddings';

export interface EmbeddingProvider {
  readonly name: string;
  readonly version: string;
  readonly dimensions: number;

  initialize(): Promise<void>;
  generateEmbedding(text: string): Promise<number[]>;
  isAvailable(): Promise<boolean>;
}

export const MODEL_DIMENSIONS: Record<string, number> = {
  'minilm-l6-v2': 384,
  'mpnet-base-v2': 768,
};

export function getEmbeddingDimensions(version: string): number {
  return MODEL_DIMENSIONS[version] || 384;
}

export function isLegacyEmbedding(data: EmbeddingData): boolean {
  return !data.version;
}

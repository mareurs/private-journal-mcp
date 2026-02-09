// ABOUTME: Embedding provider wrapping @xenova/transformers (existing default backend)
// ABOUTME: Uses all-MiniLM-L6-v2 model for 384-dim embeddings on CPU

import { pipeline, FeatureExtractionPipeline } from '@xenova/transformers';
import { EmbeddingProvider } from './providers';

export class TransformersProvider implements EmbeddingProvider {
  readonly name = 'transformers.js';
  readonly version = 'minilm-l6-v2';
  readonly dimensions = 384;

  private extractor: FeatureExtractionPipeline | null = null;
  private readonly modelName = 'Xenova/all-MiniLM-L6-v2';

  async initialize(): Promise<void> {
    console.error('Loading embedding model (transformers.js)...');
    this.extractor = await pipeline('feature-extraction', this.modelName);
    console.error('Embedding model loaded successfully');
  }

  async generateEmbedding(text: string): Promise<number[]> {
    if (!this.extractor) {
      await this.initialize();
    }
    if (!this.extractor) {
      throw new Error('Embedding model not initialized');
    }
    const result = await this.extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(result.data);
  }

  async isAvailable(): Promise<boolean> {
    try {
      require.resolve('@xenova/transformers');
      return true;
    } catch {
      return false;
    }
  }
}

// ABOUTME: Tests for embedding provider interface and version-aware embedding data
// ABOUTME: Validates provider contract and backward compatibility with legacy embeddings

import { EmbeddingProvider, getEmbeddingDimensions, isLegacyEmbedding } from '../src/providers';
import { TransformersProvider } from '../src/transformers-provider';
import { EmbeddingData } from '../src/embeddings';

describe('TransformersProvider', () => {
  test('has correct metadata', () => {
    const provider = new TransformersProvider();
    expect(provider.name).toBe('transformers.js');
    expect(provider.version).toBe('minilm-l6-v2');
    expect(provider.dimensions).toBe(384);
  });

  test('reports as available', async () => {
    const provider = new TransformersProvider();
    const available = await provider.isAvailable();
    expect(available).toBe(true);
  });
});

describe('EmbeddingProvider interface', () => {
  test('legacy embeddings without version field are detected', () => {
    const legacy: EmbeddingData = {
      embedding: new Array(384).fill(0.1),
      text: 'test',
      sections: ['Feelings'],
      timestamp: Date.now(),
      path: '/tmp/test.md'
    };

    expect(isLegacyEmbedding(legacy)).toBe(true);
  });

  test('versioned embeddings are not legacy', () => {
    const versioned: EmbeddingData = {
      embedding: new Array(768).fill(0.1),
      text: 'test',
      sections: ['Feelings'],
      timestamp: Date.now(),
      path: '/tmp/test.md',
      version: 'mpnet-base-v2',
      dimensions: 768
    };

    expect(isLegacyEmbedding(versioned)).toBe(false);
  });

  test('getEmbeddingDimensions returns correct values', () => {
    expect(getEmbeddingDimensions('minilm-l6-v2')).toBe(384);
    expect(getEmbeddingDimensions('mpnet-base-v2')).toBe(768);
  });
});

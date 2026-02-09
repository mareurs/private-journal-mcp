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

describe('OnnxProvider', () => {
  test('has correct metadata', async () => {
    const { OnnxProvider } = await import('../src/onnx-provider');
    const provider = new OnnxProvider();
    expect(provider.name).toBe('onnxruntime');
    expect(provider.version).toBe('mpnet-base-v2');
    expect(provider.dimensions).toBe(768);
  });

  test('reports availability based on onnxruntime-node', async () => {
    const { OnnxProvider } = await import('../src/onnx-provider');
    const provider = new OnnxProvider();
    const available = await provider.isAvailable();
    expect(typeof available).toBe('boolean');
  });

  test('generates 768-dimensional embeddings', async () => {
    const { OnnxProvider } = await import('../src/onnx-provider');
    const provider = new OnnxProvider();

    if (!(await provider.isAvailable())) {
      console.warn('Skipping: onnxruntime-node not available');
      return;
    }

    await provider.initialize();
    const embedding = await provider.generateEmbedding('Test text about programming');

    expect(embedding).toHaveLength(768);

    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    expect(norm).toBeCloseTo(1.0, 3);
  }, 120000);
});

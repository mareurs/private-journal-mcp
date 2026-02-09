// ABOUTME: Local embedding service using pluggable providers for semantic journal search
// ABOUTME: Provides text embedding generation and similarity computation utilities

import * as fs from 'fs/promises';
import { EmbeddingProvider } from './providers';
import { TransformersProvider } from './transformers-provider';

export interface EmbeddingData {
  embedding: number[];
  text: string;
  sections: string[];
  timestamp: number;
  path: string;
  version?: string;
  dimensions?: number;
}

export class EmbeddingService {
  private static instance: EmbeddingService;
  private provider: EmbeddingProvider | null = null;
  private initPromise: Promise<void> | null = null;

  private constructor() {}

  static getInstance(): EmbeddingService {
    if (!EmbeddingService.instance) {
      EmbeddingService.instance = new EmbeddingService();
    }
    return EmbeddingService.instance;
  }

  get currentVersion(): string {
    return this.provider?.version || 'minilm-l6-v2';
  }

  get currentDimensions(): number {
    return this.provider?.dimensions || 384;
  }

  async initialize(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }
    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    this.provider = await this.selectProvider();
    await this.provider.initialize();
  }

  private async selectProvider(): Promise<EmbeddingProvider> {
    // Try ONNX provider first (if available)
    try {
      // @ts-ignore - onnx-provider is optional and may not exist yet
      const { OnnxProvider } = await import('./onnx-provider');
      const onnx = new OnnxProvider();
      if (await onnx.isAvailable()) {
        console.error('ONNX Runtime detected, using GPU-accelerated embeddings');
        return onnx;
      }
    } catch {
      // onnxruntime-node not installed, fall through
    }

    // Default to transformers.js
    console.error('Using default transformers.js embeddings');
    return new TransformersProvider();
  }

  async generateEmbedding(text: string): Promise<number[]> {
    if (!this.provider) {
      await this.initialize();
    }
    if (!this.provider) {
      throw new Error('Embedding provider not initialized');
    }
    return this.provider.generateEmbedding(text);
  }

  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same length');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  async saveEmbedding(filePath: string, embeddingData: EmbeddingData): Promise<void> {
    const embeddingPath = filePath.replace(/\.md$/, '.embedding');
    await fs.writeFile(embeddingPath, JSON.stringify(embeddingData, null, 2), 'utf8');
  }

  async loadEmbedding(filePath: string): Promise<EmbeddingData | null> {
    const embeddingPath = filePath.replace(/\.md$/, '.embedding');

    try {
      const content = await fs.readFile(embeddingPath, 'utf8');
      return JSON.parse(content);
    } catch (error) {
      if ((error as any)?.code === 'ENOENT') {
        return null;
      }
      throw error;
    }
  }

  extractSearchableText(markdownContent: string): { text: string; sections: string[] } {
    const withoutFrontmatter = markdownContent.replace(/^---\n.*?\n---\n/s, '');

    const sections: string[] = [];
    const sectionMatches = withoutFrontmatter.match(/^## (.+)$/gm);
    if (sectionMatches) {
      sections.push(...sectionMatches.map(match => match.replace('## ', '')));
    }

    const cleanText = withoutFrontmatter
      .replace(/^## .+$/gm, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    return { text: cleanText, sections };
  }
}

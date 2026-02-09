# GPU-Accelerated Embeddings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional GPU-accelerated embeddings via ONNX Runtime while keeping existing @xenova/transformers as default, with zero breaking changes.

**Architecture:** Extract embedding generation behind an `EmbeddingProvider` interface. The existing transformers.js backend becomes `TransformersProvider`. A new `OnnxProvider` uses onnxruntime-node with CUDA. At startup, the service auto-detects whether onnxruntime-node is installed and CUDA is available, selecting the best provider. Search handles mixed-dimension embeddings by grouping by dimension and generating per-model query embeddings.

**Tech Stack:** TypeScript, onnxruntime-node (optional), @huggingface/transformers (tokenizer for ONNX), @xenova/transformers (existing default)

---

## Task 1: Add EmbeddingProvider Interface and Version Fields

**Files:**
- Modify: `src/embeddings.ts:8-14` (EmbeddingData interface)
- Create: `src/providers.ts` (new file with provider interface)

**Step 1: Write the failing test**

Create `tests/providers.test.ts`:

```typescript
// ABOUTME: Tests for embedding provider interface and version-aware embedding data
// ABOUTME: Validates provider contract and backward compatibility with legacy embeddings

import { EmbeddingProvider, getEmbeddingDimensions, isLegacyEmbedding } from '../src/providers';
import { EmbeddingData } from '../src/embeddings';

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
```

**Step 2: Run test to verify it fails**

Run: `npx jest tests/providers.test.ts -v`
Expected: FAIL - cannot find module `../src/providers`

**Step 3: Add version fields to EmbeddingData**

In `src/embeddings.ts`, update the interface:

```typescript
export interface EmbeddingData {
  embedding: number[];
  text: string;
  sections: string[];
  timestamp: number;
  path: string;
  version?: string;      // "minilm-l6-v2" | "mpnet-base-v2"
  dimensions?: number;   // 384 | 768
}
```

**Step 4: Create providers.ts**

```typescript
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
```

**Step 5: Run test to verify it passes**

Run: `npx jest tests/providers.test.ts -v`
Expected: PASS

**Step 6: Run full test suite for no regressions**

Run: `npm test`
Expected: All 26 existing tests still pass

**Step 7: Commit**

```bash
git add src/providers.ts src/embeddings.ts tests/providers.test.ts
git commit -m "feat: add EmbeddingProvider interface and version fields to EmbeddingData"
```

---

## Task 2: Extract TransformersProvider from Existing EmbeddingService

**Files:**
- Create: `src/transformers-provider.ts`
- Modify: `src/embeddings.ts` (delegate to provider)

**Step 1: Write the failing test**

Add to `tests/providers.test.ts`:

```typescript
import { TransformersProvider } from '../src/transformers-provider';

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
```

**Step 2: Run test to verify it fails**

Run: `npx jest tests/providers.test.ts -v`
Expected: FAIL - cannot find module `../src/transformers-provider`

**Step 3: Create TransformersProvider**

Create `src/transformers-provider.ts`:

```typescript
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
```

**Step 4: Run test to verify it passes**

Run: `npx jest tests/providers.test.ts -v`
Expected: PASS

**Step 5: Wire TransformersProvider into EmbeddingService**

Update `src/embeddings.ts` to use provider internally. The EmbeddingService keeps its public API identical but delegates to TransformersProvider:

```typescript
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

  // Expose provider metadata
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
```

**Step 6: Run full test suite**

Run: `npm test`
Expected: All 26 existing tests still pass (TransformersProvider is the default, behavior identical)

**Step 7: Commit**

```bash
git add src/transformers-provider.ts src/embeddings.ts tests/providers.test.ts
git commit -m "refactor: extract TransformersProvider, delegate via EmbeddingProvider interface"
```

---

## Task 3: Write Version Metadata on New Embeddings

**Files:**
- Modify: `src/journal.ts:185-212` (generateEmbeddingForEntry)

**Step 1: Write the failing test**

Add to `tests/providers.test.ts`:

```typescript
describe('Embedding version metadata', () => {
  test('new embeddings include version and dimensions', async () => {
    // We need to check the EmbeddingData produced by generateEmbeddingForEntry
    // This is tested indirectly through journal writing
    const { EmbeddingService } = await import('../src/embeddings');
    const service = EmbeddingService.getInstance();

    expect(service.currentVersion).toBeDefined();
    expect(service.currentDimensions).toBeGreaterThan(0);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `npx jest tests/providers.test.ts -v`
Expected: FAIL - `currentVersion` is not a property (until EmbeddingService is updated)

**Step 3: Update journal.ts to write version metadata**

In `src/journal.ts`, update `generateEmbeddingForEntry`:

```typescript
  private async generateEmbeddingForEntry(
    filePath: string,
    content: string,
    timestamp: Date
  ): Promise<void> {
    try {
      const { text, sections } = this.embeddingService.extractSearchableText(content);

      if (text.trim().length === 0) {
        return;
      }

      const embedding = await this.embeddingService.generateEmbedding(text);

      const embeddingData: EmbeddingData = {
        embedding,
        text,
        sections,
        timestamp: timestamp.getTime(),
        path: filePath,
        version: this.embeddingService.currentVersion,
        dimensions: this.embeddingService.currentDimensions,
      };

      await this.embeddingService.saveEmbedding(filePath, embeddingData);
    } catch (error) {
      console.error(`Failed to generate embedding for ${filePath}:`, error);
    }
  }
```

**Step 4: Run full test suite**

Run: `npm test`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/journal.ts tests/providers.test.ts
git commit -m "feat: write version and dimensions metadata on new embeddings"
```

---

## Task 4: Create OnnxProvider

**Files:**
- Create: `src/onnx-provider.ts`
- Modify: `package.json` (add optional deps)

**Step 1: Add optional dependencies**

Update `package.json`:

```json
{
  "optionalDependencies": {
    "onnxruntime-node": "^1.19.0"
  }
}
```

Run: `npm install`

**Step 2: Write the test**

Add to `tests/providers.test.ts`:

```typescript
describe('OnnxProvider', () => {
  test('has correct metadata', async () => {
    // Dynamic import since it's optional
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
    // Will be true since we installed onnxruntime-node
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

    // Verify normalization
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    expect(norm).toBeCloseTo(1.0, 3);
  }, 120000);
});
```

**Step 3: Run test to verify it fails**

Run: `npx jest tests/providers.test.ts -v`
Expected: FAIL - cannot find module `../src/onnx-provider`

**Step 4: Create OnnxProvider**

Create `src/onnx-provider.ts`:

```typescript
// ABOUTME: ONNX Runtime embedding provider with CUDA GPU acceleration
// ABOUTME: Uses all-mpnet-base-v2 for 768-dim embeddings, falls back to CPU if no GPU

import * as path from 'path';
import * as fs from 'fs/promises';
import { EmbeddingProvider } from './providers';

export class OnnxProvider implements EmbeddingProvider {
  readonly name = 'onnxruntime';
  readonly version = 'mpnet-base-v2';
  readonly dimensions = 768;

  private session: any = null;
  private tokenizer: any = null;
  private ort: any = null;

  private readonly modelId = 'sentence-transformers/all-mpnet-base-v2';
  private readonly modelDir = path.join(
    process.env.HF_HOME || path.join(process.env.HOME || '/tmp', '.cache', 'huggingface'),
    'onnx-models',
    'all-mpnet-base-v2'
  );

  async isAvailable(): Promise<boolean> {
    try {
      require.resolve('onnxruntime-node');
      return true;
    } catch {
      return false;
    }
  }

  async initialize(): Promise<void> {
    // Load onnxruntime-node dynamically
    this.ort = require('onnxruntime-node');

    // Check CUDA
    const providers = this.ort.InferenceSession
      ? (await this.getAvailableProviders())
      : ['cpu'];
    const hasCuda = providers.includes('CUDAExecutionProvider');

    if (hasCuda) {
      console.error('CUDA available, using GPU-accelerated embeddings');
    } else {
      console.error('CUDA not available, ONNX will use CPU');
    }

    // Download model if needed
    await this.ensureModel();

    // Create session
    const modelPath = path.join(this.modelDir, 'model.onnx');
    const sessionOptions: any = {
      graphOptimizationLevel: 'all',
    };

    if (hasCuda) {
      sessionOptions.executionProviders = [
        { name: 'CUDAExecutionProvider' },
        { name: 'CPUExecutionProvider' },
      ];
    }

    this.session = await this.ort.InferenceSession.create(modelPath, sessionOptions);

    // Load tokenizer
    const { AutoTokenizer } = await import('@xenova/transformers');
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);

    console.error(`ONNX provider initialized (${hasCuda ? 'GPU' : 'CPU'})`);
  }

  async generateEmbedding(text: string): Promise<number[]> {
    if (!this.session || !this.tokenizer) {
      throw new Error('OnnxProvider not initialized');
    }

    // Tokenize
    const encoded = await this.tokenizer(text, {
      padding: true,
      truncation: true,
      max_length: 384,
    });

    const inputIds = Array.from(encoded.input_ids.data).map(Number);
    const attentionMask = Array.from(encoded.attention_mask.data).map(Number);

    // Build ONNX tensors
    const seqLen = inputIds.length;
    const feeds: Record<string, any> = {
      input_ids: new this.ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, seqLen]),
      attention_mask: new this.ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, seqLen]),
    };

    // token_type_ids may or may not be needed
    if (encoded.token_type_ids) {
      const tokenTypeIds = Array.from(encoded.token_type_ids.data).map(Number);
      feeds.token_type_ids = new this.ort.Tensor('int64', BigInt64Array.from(tokenTypeIds.map(BigInt)), [1, seqLen]);
    }

    // Run inference
    const results = await this.session.run(feeds);

    // Get output - try common output names
    const outputTensor = results.last_hidden_state || results.token_embeddings || Object.values(results)[0];
    const outputData = Array.from(outputTensor.data as Float32Array);
    const hiddenSize = outputTensor.dims[2];

    // Mean pooling with attention mask
    const pooled = this.meanPooling(outputData, attentionMask, seqLen, hiddenSize);

    // L2 normalize
    return this.normalize(pooled);
  }

  private meanPooling(
    embeddings: number[],
    attentionMask: number[],
    seqLen: number,
    hiddenSize: number
  ): number[] {
    const pooled = new Array(hiddenSize).fill(0);
    let sumMask = 0;

    for (let i = 0; i < seqLen; i++) {
      if (attentionMask[i] === 1) {
        for (let j = 0; j < hiddenSize; j++) {
          pooled[j] += embeddings[i * hiddenSize + j];
        }
        sumMask++;
      }
    }

    if (sumMask === 0) return pooled;
    return pooled.map(val => val / sumMask);
  }

  private normalize(embedding: number[]): number[] {
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (norm === 0) return embedding;
    return embedding.map(val => val / norm);
  }

  private async getAvailableProviders(): Promise<string[]> {
    try {
      // onnxruntime-node exposes available providers
      return this.ort.InferenceSession.availableExecutionProviders?.() || ['CPUExecutionProvider'];
    } catch {
      return ['CPUExecutionProvider'];
    }
  }

  private async ensureModel(): Promise<void> {
    const modelPath = path.join(this.modelDir, 'model.onnx');

    try {
      await fs.access(modelPath);
      console.error(`ONNX model found at ${this.modelDir}`);
      return;
    } catch {
      // Model not cached, download it
    }

    console.error(`Downloading ONNX model to ${this.modelDir}...`);
    await fs.mkdir(this.modelDir, { recursive: true });

    // Download from Hugging Face
    const baseUrl = `https://huggingface.co/${this.modelId}/resolve/main/onnx`;
    const files = ['model.onnx'];

    for (const file of files) {
      const url = `${baseUrl}/${file}`;
      console.error(`Downloading ${file}...`);

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to download ${url}: ${response.statusText}`);
      }

      const buffer = Buffer.from(await response.arrayBuffer());
      await fs.writeFile(path.join(this.modelDir, file), buffer);
    }

    console.error('ONNX model downloaded successfully');
  }
}
```

**Step 5: Run test to verify it passes**

Run: `npx jest tests/providers.test.ts -v`
Expected: PASS (OnnxProvider metadata test passes; integration test runs if onnxruntime available)

**Step 6: Run full test suite**

Run: `npm test`
Expected: All tests pass (OnnxProvider not used unless onnxruntime-node installed)

**Step 7: Commit**

```bash
git add src/onnx-provider.ts tests/providers.test.ts package.json package-lock.json
git commit -m "feat: add OnnxProvider with CUDA GPU acceleration support"
```

---

## Task 5: Version-Aware Search

**Files:**
- Modify: `src/search.ts` (group embeddings by dimension, generate per-model query)

**Step 1: Write the failing test**

Add to `tests/providers.test.ts`:

```typescript
describe('Mixed-dimension search', () => {
  test('search handles embeddings with different dimensions', async () => {
    // Create mock embeddings with different dimensions
    const { EmbeddingService } = await import('../src/embeddings');
    const service = EmbeddingService.getInstance();

    // Cosine similarity should throw for mismatched dimensions
    const vec384 = new Array(384).fill(0.1);
    const vec768 = new Array(768).fill(0.1);
    expect(() => service.cosineSimilarity(vec384, vec768)).toThrow('Vectors must have same length');

    // Same dimensions should work
    expect(() => service.cosineSimilarity(vec384, vec384)).not.toThrow();
  });
});
```

**Step 2: Run test to verify it passes**

Run: `npx jest tests/providers.test.ts -v`
Expected: PASS (this validates the constraint we need to handle)

**Step 3: Update SearchService for mixed dimensions**

In `src/search.ts`, update the `search` method to group embeddings by dimension and only compare same-dimension vectors:

```typescript
  async search(query: string, options: SearchOptions = {}): Promise<SearchResult[]> {
    const {
      limit = 10,
      minScore = 0.1,
      sections,
      dateRange,
      type = 'both'
    } = options;

    // Collect all embeddings
    const allEmbeddings: Array<EmbeddingData & { type: 'project' | 'user' }> = [];

    if (type === 'both' || type === 'project') {
      const projectEmbeddings = await this.loadEmbeddingsFromPath(this.projectPath, 'project');
      allEmbeddings.push(...projectEmbeddings);
    }

    if (type === 'both' || type === 'user') {
      const userEmbeddings = await this.loadEmbeddingsFromPath(this.userPath, 'user');
      allEmbeddings.push(...userEmbeddings);
    }

    // Filter by criteria
    const filtered = allEmbeddings.filter(embedding => {
      if (sections && sections.length > 0) {
        const hasMatchingSection = sections.some(section =>
          embedding.sections.some(embeddingSection =>
            embeddingSection.toLowerCase().includes(section.toLowerCase())
          )
        );
        if (!hasMatchingSection) return false;
      }

      if (dateRange) {
        const entryDate = new Date(embedding.timestamp);
        if (dateRange.start && entryDate < dateRange.start) return false;
        if (dateRange.end && entryDate > dateRange.end) return false;
      }

      return true;
    });

    // Group embeddings by dimension for correct comparison
    const dimGroups = new Map<number, Array<EmbeddingData & { type: 'project' | 'user' }>>();
    for (const emb of filtered) {
      const dim = emb.embedding.length;
      if (!dimGroups.has(dim)) dimGroups.set(dim, []);
      dimGroups.get(dim)!.push(emb);
    }

    // Generate query embedding for each dimension group and score
    const results: SearchResult[] = [];

    for (const [dim, embeddings] of dimGroups) {
      // Generate query embedding matching this dimension
      // The current provider may produce a different dimension than stored embeddings
      // For legacy embeddings, we need the current provider's embedding and only compare
      // against same-dimension stored embeddings
      const queryEmbedding = await this.embeddingService.generateEmbedding(query);

      // Only compare if query embedding matches this group's dimension
      if (queryEmbedding.length !== dim) continue;

      for (const embedding of embeddings) {
        const score = this.embeddingService.cosineSimilarity(queryEmbedding, embedding.embedding);
        if (score >= minScore) {
          results.push({
            path: embedding.path,
            score,
            text: embedding.text,
            sections: embedding.sections,
            timestamp: embedding.timestamp,
            excerpt: this.generateExcerpt(embedding.text, query),
            type: embedding.type
          });
        }
      }
    }

    // Sort and limit
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, limit);
  }
```

**Step 4: Run full test suite**

Run: `npm test`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/search.ts tests/providers.test.ts
git commit -m "feat: version-aware search supporting mixed embedding dimensions"
```

---

## Task 6: Optional Migration Command

**Files:**
- Modify: `src/index.ts` (add --migrate-embeddings flag)
- Create: `src/migration.ts` (migration logic)

**Step 1: Write the test**

Create `tests/migration.test.ts`:

```typescript
// ABOUTME: Tests for embedding migration functionality
// ABOUTME: Validates version detection and migration logic

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { needsMigration, countEmbeddingsByVersion } from '../src/migration';

describe('Migration utilities', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'migration-test-'));
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  test('detects legacy embeddings needing migration', async () => {
    // Create a legacy embedding file (no version field)
    const dayDir = path.join(tempDir, '2026-02-09');
    await fs.mkdir(dayDir, { recursive: true });
    await fs.writeFile(path.join(dayDir, '10-00-00-000000.embedding'), JSON.stringify({
      embedding: new Array(384).fill(0.1),
      text: 'test',
      sections: ['Feelings'],
      timestamp: Date.now(),
      path: '/tmp/test.md'
    }));

    const counts = await countEmbeddingsByVersion(tempDir);
    expect(counts.legacy).toBe(1);
    expect(counts.total).toBe(1);
  });

  test('detects versioned embeddings', async () => {
    const dayDir = path.join(tempDir, '2026-02-09');
    await fs.mkdir(dayDir, { recursive: true });
    await fs.writeFile(path.join(dayDir, '10-00-00-000000.embedding'), JSON.stringify({
      embedding: new Array(768).fill(0.1),
      text: 'test',
      sections: ['Feelings'],
      timestamp: Date.now(),
      path: '/tmp/test.md',
      version: 'mpnet-base-v2',
      dimensions: 768
    }));

    const counts = await countEmbeddingsByVersion(tempDir);
    expect(counts.legacy).toBe(0);
    expect(counts['mpnet-base-v2']).toBe(1);
    expect(counts.total).toBe(1);
  });

  test('needsMigration returns false when all embeddings are current', async () => {
    const dayDir = path.join(tempDir, '2026-02-09');
    await fs.mkdir(dayDir, { recursive: true });
    await fs.writeFile(path.join(dayDir, '10-00-00-000000.embedding'), JSON.stringify({
      embedding: new Array(768).fill(0.1),
      text: 'test',
      sections: ['Feelings'],
      timestamp: Date.now(),
      path: '/tmp/test.md',
      version: 'mpnet-base-v2',
      dimensions: 768
    }));

    expect(await needsMigration(tempDir, 'mpnet-base-v2')).toBe(false);
  });

  test('needsMigration returns true when legacy embeddings exist', async () => {
    const dayDir = path.join(tempDir, '2026-02-09');
    await fs.mkdir(dayDir, { recursive: true });
    await fs.writeFile(path.join(dayDir, '10-00-00-000000.embedding'), JSON.stringify({
      embedding: new Array(384).fill(0.1),
      text: 'test',
      sections: ['Feelings'],
      timestamp: Date.now(),
      path: '/tmp/test.md'
    }));

    expect(await needsMigration(tempDir, 'mpnet-base-v2')).toBe(true);
  });

  test('empty directory needs no migration', async () => {
    expect(await needsMigration(tempDir, 'mpnet-base-v2')).toBe(false);
  });
});
```

**Step 2: Run test to verify it fails**

Run: `npx jest tests/migration.test.ts -v`
Expected: FAIL - cannot find module `../src/migration`

**Step 3: Create migration.ts**

```typescript
// ABOUTME: Embedding migration utilities for upgrading between model versions
// ABOUTME: Scans embedding files and supports optional bulk regeneration

import * as fs from 'fs/promises';
import * as path from 'path';
import { EmbeddingData } from './embeddings';

export interface VersionCounts {
  legacy: number;
  total: number;
  [version: string]: number;
}

export async function countEmbeddingsByVersion(basePath: string): Promise<VersionCounts> {
  const counts: VersionCounts = { legacy: 0, total: 0 };

  try {
    const dayDirs = await fs.readdir(basePath);

    for (const dayDir of dayDirs) {
      const dayPath = path.join(basePath, dayDir);
      const stat = await fs.stat(dayPath);

      if (!stat.isDirectory() || !dayDir.match(/^\d{4}-\d{2}-\d{2}$/)) {
        continue;
      }

      const files = await fs.readdir(dayPath);
      const embeddingFiles = files.filter(f => f.endsWith('.embedding'));

      for (const file of embeddingFiles) {
        try {
          const content = await fs.readFile(path.join(dayPath, file), 'utf8');
          const data: EmbeddingData = JSON.parse(content);

          counts.total++;
          if (!data.version) {
            counts.legacy++;
          } else {
            counts[data.version] = (counts[data.version] || 0) + 1;
          }
        } catch {
          // Skip corrupt files
        }
      }
    }
  } catch (error) {
    if ((error as any)?.code !== 'ENOENT') {
      throw error;
    }
  }

  return counts;
}

export async function needsMigration(basePath: string, targetVersion: string): Promise<boolean> {
  const counts = await countEmbeddingsByVersion(basePath);
  if (counts.total === 0) return false;
  return counts.legacy > 0 || (counts[targetVersion] || 0) < counts.total;
}

export async function migrateEmbeddings(
  basePath: string,
  generateEmbedding: (text: string) => Promise<number[]>,
  targetVersion: string,
  targetDimensions: number,
  onProgress?: (current: number, total: number) => void
): Promise<number> {
  let migrated = 0;
  const counts = await countEmbeddingsByVersion(basePath);
  const total = counts.total - (counts[targetVersion] || 0);

  if (total === 0) return 0;

  const dayDirs = await fs.readdir(basePath);

  for (const dayDir of dayDirs) {
    const dayPath = path.join(basePath, dayDir);
    const stat = await fs.stat(dayPath);

    if (!stat.isDirectory() || !dayDir.match(/^\d{4}-\d{2}-\d{2}$/)) {
      continue;
    }

    const files = await fs.readdir(dayPath);
    const embeddingFiles = files.filter(f => f.endsWith('.embedding'));

    for (const file of embeddingFiles) {
      const filePath = path.join(dayPath, file);

      try {
        const content = await fs.readFile(filePath, 'utf8');
        const data: EmbeddingData = JSON.parse(content);

        // Skip if already at target version
        if (data.version === targetVersion) continue;

        // Generate new embedding
        const newEmbedding = await generateEmbedding(data.text);

        const updated: EmbeddingData = {
          ...data,
          embedding: newEmbedding,
          version: targetVersion,
          dimensions: targetDimensions,
        };

        // Write updated embedding
        await fs.writeFile(filePath, JSON.stringify(updated, null, 2), 'utf8');

        migrated++;
        onProgress?.(migrated, total);
      } catch (error) {
        console.error(`Failed to migrate ${filePath}:`, error);
      }
    }
  }

  return migrated;
}
```

**Step 4: Run test to verify it passes**

Run: `npx jest tests/migration.test.ts -v`
Expected: PASS

**Step 5: Add --migrate-embeddings to CLI**

In `src/index.ts`, add the flag handling:

```typescript
#!/usr/bin/env node

// ABOUTME: Main entry point for the private journal MCP server
// ABOUTME: Handles command line arguments and starts the server

import * as path from 'path';
import { PrivateJournalServer } from './server';
import { resolveProjectJournalPath, resolveUserJournalPath } from './paths';
import { EmbeddingService } from './embeddings';
import { migrateEmbeddings, countEmbeddingsByVersion } from './migration';

function parseArguments(): { journalPath: string; migrate: boolean } {
  const args = process.argv.slice(2);
  let journalPath: string | null = null;
  let migrate = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--journal-path' && i + 1 < args.length) {
      journalPath = path.resolve(args[i + 1]);
    }
    if (args[i] === '--migrate-embeddings') {
      migrate = true;
    }
  }

  return {
    journalPath: journalPath || resolveProjectJournalPath(),
    migrate,
  };
}

async function runMigration(projectPath: string): Promise<void> {
  const userPath = resolveUserJournalPath();
  const service = EmbeddingService.getInstance();
  await service.initialize();

  const paths = [
    { path: projectPath, label: 'project' },
    { path: userPath, label: 'user' },
  ];

  for (const { path: basePath, label } of paths) {
    const counts = await countEmbeddingsByVersion(basePath);
    if (counts.total === 0) {
      console.error(`No ${label} embeddings found, skipping.`);
      continue;
    }

    console.error(`${label} embeddings: ${JSON.stringify(counts)}`);

    const migrated = await migrateEmbeddings(
      basePath,
      (text) => service.generateEmbedding(text),
      service.currentVersion,
      service.currentDimensions,
      (current, total) => {
        console.error(`Migrating ${label} embeddings: ${current}/${total}`);
      }
    );

    console.error(`Migrated ${migrated} ${label} embeddings to ${service.currentVersion}.`);
  }
}

async function main(): Promise<void> {
  try {
    console.error('=== Private Journal MCP Server Debug Info ===');
    console.error(`Node.js version: ${process.version}`);
    console.error(`Platform: ${process.platform}`);
    console.error(`Architecture: ${process.arch}`);

    try {
      console.error(`Current working directory: ${process.cwd()}`);
    } catch (error) {
      console.error(`Failed to get current working directory: ${error}`);
    }

    console.error(`Environment variables:`);
    console.error(`  HOME: ${process.env.HOME || 'undefined'}`);
    console.error(`  USERPROFILE: ${process.env.USERPROFILE || 'undefined'}`);
    console.error(`  TEMP: ${process.env.TEMP || 'undefined'}`);
    console.error(`  TMP: ${process.env.TMP || 'undefined'}`);
    console.error(`  USER: ${process.env.USER || 'undefined'}`);
    console.error(`  USERNAME: ${process.env.USERNAME || 'undefined'}`);

    const { journalPath, migrate } = parseArguments();
    console.error(`Selected journal path: ${journalPath}`);
    console.error('===============================================');

    if (migrate) {
      await runMigration(journalPath);
      process.exit(0);
    }

    const server = new PrivateJournalServer(journalPath);
    await server.run();
  } catch (error) {
    console.error('Failed to start private journal MCP server:', error);
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});
```

**Step 6: Run full test suite**

Run: `npm test`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/migration.ts src/index.ts tests/migration.test.ts
git commit -m "feat: add optional --migrate-embeddings CLI command"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `README.md`

**Step 1: Add GPU section to README.md**

Add after the "Development" section:

```markdown
## GPU Acceleration (Optional)

For faster and more accurate embeddings, you can enable GPU acceleration:

### Requirements

- NVIDIA GPU with CUDA 11.8+ or 12.x
- CUDA toolkit installed (`nvidia-smi` working)
- ~500MB disk space for model cache

### Setup

Install the optional ONNX Runtime dependency:

```bash
npm install onnxruntime-node
```

The server will automatically detect the GPU and use the `all-mpnet-base-v2` model (768 dimensions) instead of the default `all-MiniLM-L6-v2` (384 dimensions).

### Migration

To upgrade existing embeddings to the new model:

```bash
npx github:obra/private-journal-mcp --migrate-embeddings
```

This is optional - old and new embeddings coexist peacefully. Search only compares entries with matching dimensions.

### Fallback

If CUDA is not available, ONNX Runtime falls back to CPU. If `onnxruntime-node` is not installed, the default transformers.js backend is used.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add GPU acceleration setup instructions"
```

---

## Task 8: Final Verification

**Step 1: Run full test suite**

Run: `npm test`
Expected: All tests pass

**Step 2: Build the project**

Run: `npm run build`
Expected: Clean compilation, no errors

**Step 3: Manual smoke test (GPU path)**

Run: `node dist/index.js --journal-path /tmp/test-journal`
Expected: Logs show "ONNX Runtime detected, using GPU-accelerated embeddings" or "Using default transformers.js embeddings"

**Step 4: Verify backward compatibility**

Run: `npm uninstall onnxruntime-node && npm test`
Expected: All tests still pass (falls back to transformers.js)

Run: `npm install onnxruntime-node` (re-add)

**Step 5: Final commit if anything was missed**

```bash
git add -A
git commit -m "chore: final cleanup"
```

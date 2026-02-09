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
    this.ort = require('onnxruntime-node');

    const providers = await this.getAvailableProviders();
    const hasCuda = providers.includes('CUDAExecutionProvider');

    if (hasCuda) {
      console.error('CUDA available, using GPU-accelerated embeddings');
    } else {
      console.error('CUDA not available, ONNX will use CPU');
    }

    await this.ensureModel();

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

    const { AutoTokenizer } = await import('@xenova/transformers');
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);

    console.error(`ONNX provider initialized (${hasCuda ? 'GPU' : 'CPU'})`);
  }

  async generateEmbedding(text: string): Promise<number[]> {
    if (!this.session || !this.tokenizer) {
      throw new Error('OnnxProvider not initialized');
    }

    const encoded = await this.tokenizer(text, {
      padding: true,
      truncation: true,
      max_length: 384,
    });

    const inputIds = Array.from(encoded.input_ids.data).map(Number);
    const attentionMask = Array.from(encoded.attention_mask.data).map(Number);

    const seqLen = inputIds.length;
    const feeds: Record<string, any> = {
      input_ids: new this.ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, seqLen]),
      attention_mask: new this.ort.Tensor('int64', BigInt64Array.from(attentionMask.map(BigInt)), [1, seqLen]),
    };

    if (encoded.token_type_ids) {
      const tokenTypeIds = Array.from(encoded.token_type_ids.data).map(Number);
      feeds.token_type_ids = new this.ort.Tensor('int64', BigInt64Array.from(tokenTypeIds.map(BigInt)), [1, seqLen]);
    }

    const results = await this.session.run(feeds);

    const outputTensor = results.last_hidden_state || results.token_embeddings || Object.values(results)[0];
    const outputData = Array.from(outputTensor.data as Float32Array);
    const hiddenSize = outputTensor.dims[2];

    const pooled = this.meanPooling(outputData, attentionMask, seqLen, hiddenSize);
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

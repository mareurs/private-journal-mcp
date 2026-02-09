// ABOUTME: Tests for embedding migration functionality
// ABOUTME: Validates version detection and migration logic

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { needsMigration, countEmbeddingsByVersion, migrateEmbeddings } from '../src/migration';

describe('Migration utilities', () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'migration-test-'));
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  test('detects legacy embeddings needing migration', async () => {
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

  test('migrateEmbeddings updates legacy embeddings on disk', async () => {
    const dayDir = path.join(tempDir, '2026-02-09');
    await fs.mkdir(dayDir, { recursive: true });
    const embeddingPath = path.join(dayDir, '10-00-00-000000.embedding');
    await fs.writeFile(embeddingPath, JSON.stringify({
      embedding: new Array(384).fill(0.1),
      text: 'test content',
      sections: ['Feelings'],
      timestamp: Date.now(),
      path: '/tmp/test.md'
    }));

    const mockEmbedding = new Array(768).fill(0.5);
    const generateEmbedding = jest.fn().mockResolvedValue(mockEmbedding);

    const migrated = await migrateEmbeddings(
      tempDir, generateEmbedding, 'mpnet-base-v2', 768
    );

    expect(migrated).toBe(1);
    expect(generateEmbedding).toHaveBeenCalledWith('test content');

    // Verify the file was updated on disk
    const updated = JSON.parse(await fs.readFile(embeddingPath, 'utf8'));
    expect(updated.version).toBe('mpnet-base-v2');
    expect(updated.dimensions).toBe(768);
    expect(updated.embedding).toEqual(mockEmbedding);
    expect(updated.text).toBe('test content');
  });

  test('migrateEmbeddings skips files already at target version', async () => {
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

    const generateEmbedding = jest.fn();
    const migrated = await migrateEmbeddings(
      tempDir, generateEmbedding, 'mpnet-base-v2', 768
    );

    expect(migrated).toBe(0);
    expect(generateEmbedding).not.toHaveBeenCalled();
  });

  test('migrateEmbeddings calls progress callback', async () => {
    const dayDir = path.join(tempDir, '2026-02-09');
    await fs.mkdir(dayDir, { recursive: true });
    await fs.writeFile(path.join(dayDir, '10-00-00-000000.embedding'), JSON.stringify({
      embedding: new Array(384).fill(0.1),
      text: 'first',
      sections: [],
      timestamp: Date.now(),
      path: '/tmp/first.md'
    }));
    await fs.writeFile(path.join(dayDir, '10-00-01-000000.embedding'), JSON.stringify({
      embedding: new Array(384).fill(0.1),
      text: 'second',
      sections: [],
      timestamp: Date.now(),
      path: '/tmp/second.md'
    }));

    const generateEmbedding = jest.fn().mockResolvedValue(new Array(768).fill(0.5));
    const onProgress = jest.fn();

    const migrated = await migrateEmbeddings(
      tempDir, generateEmbedding, 'mpnet-base-v2', 768, onProgress
    );

    expect(migrated).toBe(2);
    expect(onProgress).toHaveBeenCalledTimes(2);
    expect(onProgress).toHaveBeenCalledWith(1, 2);
    expect(onProgress).toHaveBeenCalledWith(2, 2);
  });
});

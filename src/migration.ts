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
  // Collect files needing migration in a single pass
  const filesToMigrate: string[] = [];

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
        const filePath = path.join(dayPath, file);
        try {
          const content = await fs.readFile(filePath, 'utf8');
          const data: EmbeddingData = JSON.parse(content);
          if (data.version !== targetVersion) {
            filesToMigrate.push(filePath);
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

  if (filesToMigrate.length === 0) return 0;

  let migrated = 0;
  const total = filesToMigrate.length;

  for (const filePath of filesToMigrate) {
    try {
      const content = await fs.readFile(filePath, 'utf8');
      const data: EmbeddingData = JSON.parse(content);

      const newEmbedding = await generateEmbedding(data.text);

      const updated: EmbeddingData = {
        ...data,
        embedding: newEmbedding,
        version: targetVersion,
        dimensions: targetDimensions,
      };

      await fs.writeFile(filePath, JSON.stringify(updated, null, 2), 'utf8');

      migrated++;
      onProgress?.(migrated, total);
    } catch (error) {
      console.error(`Failed to migrate ${filePath}:`, error);
    }
  }

  return migrated;
}

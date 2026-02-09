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
    // Log environment info for debugging
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
# Development Notes - GPU Acceleration Feature

## Implementation Decision

**Date:** 2026-02-09
**Feature:** GPU-accelerated embeddings with ONNX Runtime

## Key Constraint for PR

⚠️ **MUST NOT break existing installations**

This feature is intended as a PR to an open-source project with existing users. We cannot break their existing embeddings or require forced migration.

## Revised Implementation Strategy

### Backward-Compatible Approach

**Option B-Modified: Optional GPU Acceleration with Graceful Fallback**

1. **Keep existing `@xenova/transformers` as default**
   - Existing users continue working without changes
   - No forced migration required

2. **Add `onnxruntime-node` as optional dependency**
   - Detected automatically if installed
   - Users can `npm install onnxruntime-node` to enable GPU features

3. **Smart detection and dual-mode operation**
   - On startup, check if `onnxruntime-node` is available
   - If yes + CUDA available → use GPU-accelerated mpnet
   - If no → fall back to existing transformers.js CPU mode
   - Auto-detect embedding version from files

4. **Embedding version compatibility**
   - `.embedding` files include `version` field
   - Search works across both 384-dim and 768-dim embeddings
   - New entries use best available model
   - Old entries stay valid indefinitely

5. **Optional migration**
   - `--migrate-embeddings` CLI flag for users who want to upgrade
   - Not automatic, not required
   - Progress indicator and resumable

### Benefits of This Approach

✅ **Zero breaking changes** - existing installations work unchanged
✅ **Opt-in GPU acceleration** - users choose to install GPU deps
✅ **Mixed embeddings support** - old and new coexist peacefully
✅ **Graceful degradation** - works everywhere, optimized where possible
✅ **Easy PR review** - backward compatibility clear to maintainers

### Implementation Steps

1. Add optional ONNX Runtime support
2. Implement dual-model embedding service
3. Add version-aware search that works with both dimensions
4. Add optional migration command
5. Update docs with GPU setup instructions
6. Keep all existing tests passing
7. Add new tests for GPU mode (mocked for CI)

## Technical Details

### Dependency Structure

```json
{
  "dependencies": {
    "@xenova/transformers": "^2.17.2"  // Keep existing
  },
  "optionalDependencies": {
    "onnxruntime-node": "^1.19.0"      // New, optional
  },
  "peerDependencies": {
    "@huggingface/transformers": "^3.0.0"  // For tokenizer
  }
}
```

### Embedding Version Schema

```typescript
interface EmbeddingData {
  embedding: number[];
  text: string;
  sections: string[];
  timestamp: number;
  path: string;
  version?: string;      // "minilm-l6-v2" | "mpnet-base-v2"
  dimensions?: number;   // 384 | 768
}
```

Legacy embeddings without `version` field are treated as `minilm-l6-v2` (384-dim).

### Search Compatibility

Cosine similarity works across different dimensions - we just need to ensure we're comparing embeddings of the same dimension. Solution:

1. Generate query embedding with same model as entry
2. Or: maintain query embeddings in both models if both are active
3. Or: Use model that matches majority of existing embeddings

## Files to Modify

- `src/embeddings.ts` - Add dual-model support
- `src/types.ts` - Add version/dimension fields
- `src/search.ts` - Version-aware search
- `src/index.ts` - Add `--migrate-embeddings` flag
- `package.json` - Add optional dependencies
- `tests/embeddings.test.ts` - Add GPU mode tests (mocked)
- `README.md` - Document GPU setup as optional
- `CLAUDE.md` - Update architecture section

## Questions to Resolve

1. ~~Should we break existing installations?~~ **NO - backward compatibility required**
2. How do we handle mixed-dimension search? **Use same model as majority of existing embeddings**
3. Should migration be automatic? **NO - optional CLI flag only**
4. What's the fallback chain? **ONNX+CUDA → ONNX+CPU → transformers.js**

## Status

**Implementation complete.** All 8 tasks done, 42/42 tests passing, GPU acceleration verified on RTX A5000.

## TODO

- [ ] Test with private-journal MCP enabled across projects (both `~/.claude` and `~/.claude-sdd` profiles)
- [ ] Create PR to upstream (`obra/private-journal-mcp`) once testing is complete

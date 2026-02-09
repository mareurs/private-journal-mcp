// Global test setup
// Mock the transformers library to avoid ES module issues in Jest
jest.mock('@xenova/transformers', () => ({
  pipeline: jest.fn().mockResolvedValue(
    jest.fn().mockResolvedValue({
      data: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]) // Mock embedding vector
    })
  ),
  AutoTokenizer: {
    from_pretrained: jest.fn().mockResolvedValue(
      jest.fn().mockResolvedValue({
        input_ids: { data: BigInt64Array.from([BigInt(101), BigInt(2023), BigInt(102)]) },
        attention_mask: { data: BigInt64Array.from([BigInt(1), BigInt(1), BigInt(1)]) },
        token_type_ids: { data: BigInt64Array.from([BigInt(0), BigInt(0), BigInt(0)]) },
      })
    ),
  },
}));

// Mock onnxruntime-node to avoid needing real ONNX model files in tests
jest.mock('onnxruntime-node', () => {
  const mockOutputData = new Float32Array(3 * 768);
  for (let i = 0; i < mockOutputData.length; i++) {
    mockOutputData[i] = Math.random() * 0.1;
  }
  return {
    InferenceSession: {
      create: jest.fn().mockResolvedValue({
        run: jest.fn().mockResolvedValue({
          last_hidden_state: {
            data: mockOutputData,
            dims: [1, 3, 768],
          },
        }),
      }),
      availableExecutionProviders: jest.fn().mockReturnValue(['CPUExecutionProvider']),
    },
    Tensor: jest.fn().mockImplementation((type: string, data: any, dims: number[]) => ({
      type,
      data,
      dims,
    })),
  };
});
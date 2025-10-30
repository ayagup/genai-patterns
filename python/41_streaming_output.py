"""
Streaming Agent Pattern
Produces output incrementally as a stream
"""
import asyncio
from typing import AsyncGenerator, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import time
@dataclass
class StreamChunk:
    """A chunk of streamed output"""
    content: str
    timestamp: datetime
    chunk_index: int
    is_final: bool = False
    metadata: Dict[str, Any] = None
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
class StreamingAgent:
    """Agent that streams output incrementally"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.total_chunks_sent = 0
    async def generate_stream(self, prompt: str, chunk_size: int = 10) -> AsyncGenerator[StreamChunk, None]:
        """Generate streaming response"""
        print(f"\n{'='*60}")
        print(f"STREAMING GENERATION")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Chunk size: {chunk_size} chars\n")
        # Simulate full response
        full_response = self._generate_full_response(prompt)
        # Stream in chunks
        chunk_index = 0
        position = 0
        while position < len(full_response):
            # Get next chunk
            chunk_content = full_response[position:position + chunk_size]
            position += chunk_size
            # Create chunk
            chunk = StreamChunk(
                content=chunk_content,
                timestamp=datetime.now(),
                chunk_index=chunk_index,
                is_final=(position >= len(full_response)),
                metadata={'total_length': len(full_response)}
            )
            # Simulate processing delay
            await asyncio.sleep(0.05)
            # Yield chunk
            yield chunk
            chunk_index += 1
            self.total_chunks_sent += 1
    def _generate_full_response(self, prompt: str) -> str:
        """Generate full response (simulated)"""
        # In reality, this would call an LLM API
        if "explain" in prompt.lower():
            return (
                "Artificial Intelligence is a field of computer science that aims to create "
                "intelligent machines capable of performing tasks that typically require human "
                "intelligence. This includes learning, reasoning, problem-solving, perception, "
                "and language understanding. AI systems use various techniques including machine "
                "learning, neural networks, and natural language processing to achieve their goals."
            )
        elif "code" in prompt.lower():
            return """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")"""
        else:
            return f"This is a streaming response to: {prompt}. " * 5
    async def process_with_streaming(self, query: str) -> Dict[str, Any]:
        """Process query with streaming output"""
        print(f"Query: {query}\n")
        print("Streaming output:")
        print("-" * 60)
        full_output = ""
        chunks_received = 0
        start_time = time.time()
        first_chunk_time = None
        async for chunk in self.generate_stream(query, chunk_size=15):
            # Print chunk
            print(chunk.content, end='', flush=True)
            # Record first chunk time (time to first token)
            if first_chunk_time is None:
                first_chunk_time = time.time()
            # Accumulate
            full_output += chunk.content
            chunks_received += 1
            # Show final marker
            if chunk.is_final:
                print("\n" + "-" * 60)
                print("[STREAM COMPLETE]")
        total_time = time.time() - start_time
        time_to_first_chunk = (first_chunk_time - start_time) if first_chunk_time else 0
        return {
            'output': full_output,
            'chunks_received': chunks_received,
            'total_time_sec': total_time,
            'time_to_first_chunk_sec': time_to_first_chunk,
            'chars_per_second': len(full_output) / total_time if total_time > 0 else 0
        }
class BufferedStreamingAgent:
    """Agent that buffers and processes streaming output"""
    def __init__(self, agent_id: str, buffer_size: int = 5):
        self.agent_id = agent_id
        self.buffer_size = buffer_size
        self.buffer: List[StreamChunk] = []
    async def process_stream_with_buffer(self, stream: AsyncGenerator[StreamChunk, None]):
        """Process stream with buffering"""
        print(f"\n{'='*60}")
        print(f"BUFFERED STREAMING")
        print(f"{'='*60}")
        print(f"Buffer size: {self.buffer_size}\n")
        async for chunk in stream:
            # Add to buffer
            self.buffer.append(chunk)
            # Process when buffer is full or stream ends
            if len(self.buffer) >= self.buffer_size or chunk.is_final:
                await self._process_buffer()
    async def _process_buffer(self):
        """Process buffered chunks"""
        if not self.buffer:
            return
        # Combine buffer
        combined = ''.join(chunk.content for chunk in self.buffer)
        print(f"[Buffer] Processing {len(self.buffer)} chunks: {combined}")
        # Simulate processing
        await asyncio.sleep(0.02)
        # Clear buffer
        self.buffer.clear()
class StreamAggregator:
    """Aggregates multiple streams"""
    def __init__(self):
        self.streams: List[AsyncGenerator] = []
    def add_stream(self, stream: AsyncGenerator):
        """Add a stream to aggregate"""
        self.streams.append(stream)
    async def aggregate(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Aggregate multiple streams"""
        # Create tasks for all streams
        tasks = []
        for i, stream in enumerate(self.streams):
            tasks.append(self._consume_stream(i, stream))
        # Wait for all streams
        results = await asyncio.gather(*tasks)
        # Yield combined results
        for result in results:
            yield result
    async def _consume_stream(self, stream_id: int, stream: AsyncGenerator) -> Dict[str, Any]:
        """Consume a single stream"""
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        full_output = ''.join(chunk.content for chunk in chunks)
        return {
            'stream_id': stream_id,
            'output': full_output,
            'chunk_count': len(chunks)
        }
async def demo_basic_streaming():
    """Demonstrate basic streaming"""
    print("="*80)
    print("EXAMPLE 1: Basic Streaming")
    print("="*80)
    agent = StreamingAgent("stream-agent-001")
    result = await agent.process_with_streaming(
        "Explain artificial intelligence in simple terms"
    )
    print(f"\n{'='*60}")
    print("STREAMING STATISTICS")
    print(f"{'='*60}")
    print(f"Total chunks: {result['chunks_received']}")
    print(f"Total time: {result['total_time_sec']:.2f}s")
    print(f"Time to first chunk: {result['time_to_first_chunk_sec']:.3f}s")
    print(f"Speed: {result['chars_per_second']:.0f} chars/sec")
async def demo_buffered_streaming():
    """Demonstrate buffered streaming"""
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Buffered Streaming")
    print("="*80)
    agent = StreamingAgent("stream-agent-002")
    buffered_agent = BufferedStreamingAgent("buffer-agent-001", buffer_size=3)
    stream = agent.generate_stream("Write a Python function for calculating fibonacci", chunk_size=20)
    await buffered_agent.process_stream_with_buffer(stream)
async def demo_parallel_streaming():
    """Demonstrate parallel streaming from multiple sources"""
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Parallel Streaming")
    print("="*80)
    # Create multiple agents
    agent1 = StreamingAgent("stream-agent-003")
    agent2 = StreamingAgent("stream-agent-004")
    agent3 = StreamingAgent("stream-agent-005")
    # Create streams
    stream1 = agent1.generate_stream("Define AI", chunk_size=10)
    stream2 = agent2.generate_stream("Define ML", chunk_size=10)
    stream3 = agent3.generate_stream("Define DL", chunk_size=10)
    # Aggregate
    aggregator = StreamAggregator()
    aggregator.add_stream(stream1)
    aggregator.add_stream(stream2)
    aggregator.add_stream(stream3)
    print("Aggregating multiple streams...\n")
    async for result in aggregator.aggregate():
        print(f"\nStream {result['stream_id']}:")
        print(f"  Chunks: {result['chunk_count']}")
        print(f"  Output: {result['output'][:100]}...")
async def demo_streaming_with_callback():
    """Demonstrate streaming with callbacks"""
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Streaming with Callbacks")
    print("="*80)
    agent = StreamingAgent("stream-agent-006")
    # Callback for each chunk
    def on_chunk(chunk: StreamChunk):
        print(f"[{chunk.chunk_index}] Received: '{chunk.content}'")
    # Callback for completion
    def on_complete(full_output: str):
        print(f"\n[COMPLETE] Total length: {len(full_output)} chars")
    full_output = ""
    async for chunk in agent.generate_stream("Explain machine learning", chunk_size=12):
        on_chunk(chunk)
        full_output += chunk.content
        if chunk.is_final:
            on_complete(full_output)
async def main():
    """Run all streaming demonstrations"""
    print("="*80)
    print("STREAMING AGENT PATTERN DEMONSTRATION")
    print("="*80)
    await demo_basic_streaming()
    await demo_buffered_streaming()
    await demo_parallel_streaming()
    await demo_streaming_with_callback()
if __name__ == "__main__":
    asyncio.run(main())

"""
Main entry point for the llama-benchy CLI.
"""

import argparse
import os
import time
import uuid
import subprocess
import datetime
import numpy as np
from tabulate import tabulate
import aiohttp
import asyncio
import json
import codecs
import hashlib
from typing import Optional, Tuple
from transformers import AutoTokenizer
import requests
from pathlib import Path

# Build number is now imported from __init__.py
from . import __version__


def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Benchmark Script")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--base-url", type=str, required=True, help="OpenAI compatible endpoint URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key for the endpoint")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for benchmarking")
    parser.add_argument("--served-model-name", type=str, default=None, help="Model name used in API calls (defaults to --model if not specified)")
    parser.add_argument("--tokenizer-hf", type=str, default=None, 
                       help="Load tokenizer from HuggingFace Hub (e.g., 'gpt2', 'bert-base-uncased')")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                       help="Load tokenizer from local path (directory or file)")
    parser.add_argument("--pp", type=int, nargs='+', required=False, default=[2048], help="List of prompt processing token counts - default: 2048")
    parser.add_argument("--tg", type=int, nargs='+', required=False, default=[32], help="List of token generation counts - default: 32")
    parser.add_argument("--depth", type=int, nargs='+', default=[0], help="List of context depths (previous conversation tokens) - default: 0")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test - default: 3")
    parser.add_argument("--no-cache", action="store_true", help="Ensure unique requests to avoid prefix caching and send cache_prompt=false to the server")
    parser.add_argument("--post-run-cmd", type=str, default=None, help="Command to execute after each test run")
    parser.add_argument("--book-url", type=str, default="https://www.gutenberg.org/files/1661/1661-0.txt", help="URL of a book to use for text generation, defaults to Sherlock Holmes (https://www.gutenberg.org/files/1661/1661-0.txt)")
    parser.add_argument("--latency-mode", type=str, default="api", choices=["api", "generation", "none"], help="Method to measure latency: 'api' (list models) - default, 'generation' (single token generation), or 'none' (skip latency measurement)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase")
    parser.add_argument("--adapt-prompt", action="store_true", default=True, help="Adapt prompt size based on warmup token usage delta (default: True)")
    parser.add_argument("--no-adapt-prompt", action="store_false", dest="adapt_prompt", help="Disable prompt size adaptation")
    parser.add_argument("--enable-prefix-caching", action="store_true", help="Enable prefix caching performance measurement")
    parser.add_argument("--debug-info", action="store_true", help="Show detailed debugging information during benchmark execution")
    
    args = parser.parse_args()
    
    # Handle backward compatibility with deprecated --tokenizer argument
    if hasattr(args, 'tokenizer') and args.tokenizer:
        if args.tokenizer_hf or args.tokenizer_path:
            print("Warning: Both --tokenizer and --tokenizer-hf/--tokenizer-path specified")
            print("Ignoring deprecated --tokenizer argument")
        else:
            # Treat old --tokenizer as HuggingFace name for compatibility
            args.tokenizer_hf = args.tokenizer
            delattr(args, 'tokenizer')
    
    return args


class DummyTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


def load_tokenizer(hf_name: Optional[str] = None, local_path: Optional[str] = None):
    """
    Load tokenizer with fallback chain:
    1. HuggingFace (if hf_name specified)
    2. Local path (if local_path specified)  
    3. DummyTokenizer (fallback)
    """
    try:
        # Attempt HuggingFace loading
        if hf_name:
            return AutoTokenizer.from_pretrained(hf_name)
        
        # Attempt local path loading
        if local_path:
            path = Path(local_path)
            if path.exists():
                return AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            
        # Fall back to DummyTokenizer
        return DummyTokenizer()
        
    except Exception as e:
        print(f"Warning: Failed to load tokenizer: {e}")
        print("Using DummyTokenizer as fallback")
        return DummyTokenizer()


def prepare_text_data(book_url, tokenizer):
    try:
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "llama-benchy")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate hash of the URL for the filename
        url_hash = hashlib.md5(book_url.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{url_hash}.txt")
        
        if os.path.exists(cache_file):
            print(f"Loading text from cache: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"Downloading book from {book_url}...")
            response = requests.get(book_url)
            response.raise_for_status()
            text = response.text
            # Basic cleanup
            start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
            if start_idx != -1:
                text = text[start_idx:]
            
            # Save to cache
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved text to cache: {cache_file}")
            
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        print(f"Error downloading book: {e}")
        # Fix: Raise exception instead of exiting to allow graceful handling
        raise RuntimeError(f"Failed to download book from {book_url}: {e}")


def generate_prompt(all_tokens, tokenizer, prompt_tokens, context_tokens=0, no_cache=False):
    suffix = ""
    suffix_len = 0
    if no_cache:
        suffix = f" {uuid.uuid4()}"
        suffix_len = len(tokenizer.encode(suffix, add_special_tokens=False))
    
    # Adjust prompt tokens to fetch from text
    text_prompt_tokens = max(0, prompt_tokens - suffix_len)
    
    # Create a pool of tokens large enough
    total_needed = text_prompt_tokens + context_tokens
    
    if len(all_tokens) < total_needed:
        # Repeat tokens if not enough
        all_tokens = all_tokens * (total_needed // len(all_tokens) + 2)
    
    # Pick a random start position
    max_start = len(all_tokens) - total_needed
    start_idx = np.random.randint(0, max_start)
    
    selected_tokens = all_tokens[start_idx : start_idx + total_needed]
    
    context_text = tokenizer.decode(selected_tokens[:context_tokens]) if context_tokens > 0 else ""
    prompt_text = tokenizer.decode(selected_tokens[context_tokens:])
    
    if no_cache:
        prompt_text += suffix
        
    return context_text, prompt_text


async def measure_latency(session, base_url, api_key, mode="api", model_name=None):
    if mode == "none":
        print("Skipping latency measurement (assuming 0 ms).")
        return 0

    print(f"Measuring latency using mode: {mode}...")
    latencies = []
    headers = {"Authorization": f"Bearer {api_key}"}
    
    for _ in range(3):
        start = time.perf_counter()
        try:
            if mode == "api":
                async with session.get(f"{base_url}/models", headers=headers) as response:
                    await response.read()
                latencies.append(time.perf_counter() - start)
            elif mode == "generation":
                if not model_name:
                    raise ValueError("Model name required for generation latency mode")
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1,
                    "stream": True
                }
                async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
                    async for _ in response.content:
                        # record latency as soon as the first byte is received
                        latencies.append(time.perf_counter() - start)
                        break
                    # Drain the rest of the response to keep the connection alive
                    async for _ in response.content: pass
        except Exception as e:
            print(f"Error measuring latency: {e}")
    
    if latencies:
        avg_latency = np.mean(latencies)
        print(f"Average latency ({mode}): {avg_latency*1000:.2f} ms")
        return avg_latency
    return 0


async def warmup(session, base_url, api_key, model, tokenizer=None):
    print("Warming up...")
    headers = {"Authorization": f"Bearer {api_key}"}
    warmup_text = "Warmup " * 10
    
    delta_user = 0
    delta_context = 0
    
    # 1. User only (No Context)
    payload_user = {
        "model": model,
        "messages": [{"role": "user", "content": warmup_text}],
        "max_tokens": 1
    }
    
    try:
        async with session.post(f"{base_url}/chat/completions", json=payload_user, headers=headers) as response:
            response_json = await response.json()
            if tokenizer:
                if 'usage' in response_json:
                    prompt_tokens = response_json['usage']['prompt_tokens']
                    local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                    delta_user = prompt_tokens - local_tokens
                    print(f"Warmup (User only) complete. Delta: {delta_user} tokens (Server: {prompt_tokens}, Local: {local_tokens})")
                else:
                    print("Warmup (User only) complete (no usage stats found).")
            else:
                print("Warmup complete.")

        if tokenizer:
            # 2. System + Empty User (Context Only)
            payload_sys_empty = {
                "model": model,
                "messages": [
                    {"role": "system", "content": warmup_text},
                    {"role": "user", "content": ""}
                ],
                "max_tokens": 1
            }
            async with session.post(f"{base_url}/chat/completions", json=payload_sys_empty, headers=headers) as response:
                response_json = await response.json()
                if 'usage' in response_json:
                    prompt_tokens = response_json['usage']['prompt_tokens']
                    local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                    delta_context = prompt_tokens - local_tokens
                    print(f"Warmup (System+Empty) complete. Delta: {delta_context} tokens (Server: {prompt_tokens}, Local: {local_tokens})")
                else:
                    print("Warmup (System+Empty) complete (no usage stats found).")
                    delta_context = delta_user

    except Exception as e:
        print(f"Warmup failed: {e}")
    return delta_user, delta_context


def track_token_metrics(chunk: str, tokenizer, first_token_time: float,
                       has_first_token_tracked: bool, debug_enabled: bool) -> Tuple[int, float, bool]:
    """
    Track token metrics for a content chunk.
    
    This function calculates token counts using the provided tokenizer to maintain
    consistency with the benchmarking methodology. The token count reflects the
    tokenizer's perspective, which may differ from the actual tokens processed
    by the LLM but ensures consistent measurements across the benchmark.
    
    Args:
        chunk: The content chunk to process
        tokenizer: Tokenizer instance for encoding
        first_token_time: Current first token time
        has_first_token_tracked: Flag indicating if first token has been tracked
        debug_enabled: Whether debug logging is enabled
        
    Returns:
        tuple: (chunk_tokens, updated_first_token_time, updated_has_first_token_tracked)
               - chunk_tokens: Number of tokens in this chunk (based on tokenizer)
               - updated_first_token_time: Time when first token was detected
               - updated_has_first_token_tracked: Flag indicating first token tracking
    """
    if not chunk:
        return 0, first_token_time, has_first_token_tracked
    
    # Count tokens in this chunk using the tokenizer for consistency
    # This ensures our measurements align with the benchmarking methodology
    try:
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
    except Exception as e:
        # Handle potential encoding errors gracefully
        if debug_enabled:
            print(f"DEBUG: Error encoding chunk: {e}")
        chunk_tokens = 0
    
    # Track first token time if this is the first token-bearing chunk
    if not has_first_token_tracked and chunk_tokens > 0:
        new_first_token_time = time.perf_counter()
        if debug_enabled:
            print(f"DEBUG: First token received at {new_first_token_time:.3f}s")
        return chunk_tokens, new_first_token_time, True
    
    return chunk_tokens, first_token_time, has_first_token_tracked


async def parse_streaming_response(response_stream, tokenizer, args):
    """
    Parse streaming response and extract content/usage data.
    
    Args:
        response_stream: The streaming response object from aiohttp
        tokenizer: Tokenizer instance for token counting
        args: Arguments object containing debug_info flag
        
    Returns:
        tuple: (accumulated_content, accumulated_reasoning_content, token_count,
                prompt_usage_tokens, first_token_time, has_first_token_been_tracked)
    """
    accumulated_content = ""
    accumulated_reasoning_content = ""
    token_count = 0
    prompt_usage_tokens = 0
    first_token_time = 0
    has_first_token_been_tracked = False
    first_response_time = 0
    end_time = 0
    
    async for line in response_stream:
        line = line.decode('utf-8').strip()
        if not line:
            continue
            
        if line.startswith('data: '):
            data_str = line[6:]  # Remove 'data: ' prefix
            if data_str.strip() == '[DONE]':
                break
                
            try:
                data = json.loads(data_str)
                
                # Track first response time
                if first_response_time == 0:
                    first_response_time = time.perf_counter()
                    if args.debug_info:
                        print(f"DEBUG: First response received at {first_response_time:.3f}s")
                
                # Extract content
                if 'choices' in data and len(data['choices']) > 0:
                    choice = data['choices'][0]
                    
                    # Handle regular content
                    if 'delta' in choice and 'content' in choice['delta']:
                        content_chunk = choice['delta']['content']
                        if content_chunk:
                            accumulated_content += content_chunk
                            
                            # Use helper function to track token metrics
                            chunk_tokens, first_token_time, has_first_token_been_tracked = track_token_metrics(
                                content_chunk, tokenizer, first_token_time, has_first_token_been_tracked, args.debug_info
                            )
                            token_count += chunk_tokens
                    
                    # Handle reasoning content
                    if 'delta' in choice and 'reasoning_content' in choice['delta']:
                        reasoning_chunk = choice['delta']['reasoning_content']
                        if reasoning_chunk:
                            accumulated_reasoning_content += reasoning_chunk
                            
                            # Use helper function to track token metrics
                            chunk_tokens, first_token_time, has_first_token_been_tracked = track_token_metrics(
                                reasoning_chunk, tokenizer, first_token_time, has_first_token_been_tracked, args.debug_info
                            )
                            token_count += chunk_tokens
                    
                    # Extract usage information if present in this chunk
                    if 'usage' in data:
                        usage_data = data['usage']
                        if 'prompt_tokens' in usage_data:
                            prompt_usage_tokens = usage_data['prompt_tokens']
                        
            except json.JSONDecodeError as e:
                if args.debug_info:
                    print(f"DEBUG: JSON decode error in streaming: {e}")
                    print(f"DEBUG: Line content: {repr(line)}")
                continue
                
    end_time = time.perf_counter()
    if args.debug_info:
        print(f"DEBUG: Final token_count: {token_count}, end_time: {end_time:.3f}s")
        print(f"DEBUG: Accumulated content: {repr(accumulated_content[:100] if accumulated_content else None)}...")
        print(f"DEBUG: Accumulated reasoning_content: {repr(accumulated_reasoning_content[:100] if accumulated_reasoning_content else None)}...")
    
    return accumulated_content, accumulated_reasoning_content, token_count, prompt_usage_tokens, first_token_time, has_first_token_been_tracked


def calculate_performance_metrics(start_time, first_token_time, end_time, token_count,
                                  prompt_usage_tokens, expected_pp_tokens, latency, args):
    """
    Calculate all performance metrics from timing and token data.
    
    Args:
        start_time: When the request started
        first_token_time: When the first token was received
        end_time: When the request ended
        token_count: Total tokens generated
        prompt_usage_tokens: Actual prompt tokens used (from API)
        expected_pp_tokens: Expected prompt tokens
        latency: Measured latency
        args: Arguments object containing debug_info flag
        
    Returns:
        dict: Dictionary of calculated performance metrics
    """
    result = {
        "pp_speed": None,
        "tg_speed": None,
        "ttft": None,
        "ttfr": None,
        "est_ppt": None,
        "e2e_ttft": None
    }
    
    if token_count > 0:
        # Calculate TTFT and E2E TTFT
        e2e_ttft = first_token_time - start_time if first_token_time > 0 else 0
        ttft = e2e_ttft - latency if e2e_ttft > 0 else 0
        if ttft < 0:
            ttft = 0
            
        # Calculate token generation speed (for streaming mode)
        if first_token_time > 0 and end_time > first_token_time:
            # Generation time = total time - first token time
            generation_time = end_time - first_token_time
            if generation_time > 0:
                # Speed for generated tokens (excluding the first one which is TTFT)
                result["tg_speed"] = (token_count - 1) / generation_time if token_count > 1 else token_count / generation_time
            else:
                # Fallback if generation time is too small
                result["tg_speed"] = (token_count - 1) / 0.001 if token_count > 1 else token_count / 0.001
        else:
            # Fallback for non-streaming or edge cases
            total_request_time = end_time - start_time
            if total_request_time > 0:
                result["tg_speed"] = token_count / total_request_time
            else:
                result["tg_speed"] = token_count / 0.001  # Assume 1ms minimum
        
        # Use expected_pp_tokens for speed calculation
        total_prompt_tokens = expected_pp_tokens
        
        # Only use reported usage if it's close to expected (to handle tokenizer differences)
        # but not if it's vastly different (which happens in prefix caching where usage includes cached tokens)
        if prompt_usage_tokens > 0:
            diff = abs(prompt_usage_tokens - expected_pp_tokens)
            if diff < expected_pp_tokens * 0.2: # 20% tolerance
                 total_prompt_tokens = prompt_usage_tokens

        # Calculate TTFT (time to first token) and Estimated Prompt Processing Time
        ttfr = 0
        est_ppt = 0
        if first_token_time > 0:
            ttfr = first_token_time - start_time
            est_ppt = ttfr - latency
            if est_ppt < 0:
                est_ppt = 0

        if est_ppt > 0:
            result["pp_speed"] = total_prompt_tokens / est_ppt
            result["est_ppt"] = est_ppt

        if ttfr > 0:
            result["ttfr"] = ttfr

        if ttft > 0:
            result["ttft"] = ttft

        if e2e_ttft > 0:
            result["e2e_ttft"] = e2e_ttft
    
    return result


def execute_post_run_command(post_run_cmd):
    """
    Safely execute post-run command.
    
    Args:
        post_run_cmd: Command string to execute after benchmark run
        
    Returns:
        None (executes command and handles errors)
    """
    if post_run_cmd:
        try:
            # Fix: Avoid shell=True to prevent command injection
            subprocess.run(post_run_cmd.split(), check=True)
        except subprocess.CalledProcessError as e:
            print(f"Post-run command failed: {e}")


def setup_benchmark_payload(model_name, messages, max_tokens, no_cache, api_key):
    """
    Prepare benchmark request payload and headers.
    
    Args:
        model_name: Name of the model to benchmark
        messages: List of message dictionaries for the chat
        max_tokens: Maximum number of tokens to generate
        no_cache: Boolean indicating whether to disable caching
        api_key: API key for authentication
        
    Returns:
        tuple: (payload_dict, headers_dict)
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,  # Enable streaming for accurate token generation timing
        "stream_options": {"include_usage": True},
    }
    
    if no_cache:
        payload["cache_prompt"] = False
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    return payload, headers


async def run_benchmark(session, base_url, api_key, model_name, context_text, prompt_text, expected_pp_tokens, tg, no_cache, latency, post_run_cmd, tokenizer, args):
    """
    Orchestrates a single benchmark run by coordinating extracted functions.
    
    This refactored function follows the Single Responsibility Principle by acting
    as a coordinator for the various aspects of benchmark execution.
    
    Args:
        session: aiohttp ClientSession for HTTP requests
        base_url: Base URL for the API endpoint
        api_key: API key for authentication
        model_name: Name of the model to benchmark
        context_text: System context message
        prompt_text: User prompt message
        expected_pp_tokens: Expected prompt processing tokens
        tg: Target generation tokens
        no_cache: Whether to disable caching
        latency: Measured latency
        post_run_cmd: Command to execute after benchmark
        tokenizer: Tokenizer instance for token counting
        args: Arguments object containing configuration
        
    Returns:
        dict: Performance metrics or None if an error occurred
    """
    # Step 1: Prepare messages
    messages = []
    if context_text:
        messages.append({"role": "system", "content": context_text})
    messages.append({"role": "user", "content": prompt_text})
    
    # Initialize result structure
    result = {
        "pp_speed": None,
        "tg_speed": None,
        "ttft": None,
        "ttfr": None,
        "est_ppt": None,
        "e2e_ttft": None
    }

    try:
        # Step 2: Setup and send request
        payload, headers = setup_benchmark_payload(model_name, messages, tg, no_cache, api_key)
        start_time = time.perf_counter()

        async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"Error: {response.status} - {error_text}")
                return None
            
            # Step 3: Parse streaming response
            accumulated_content, accumulated_reasoning_content, token_count, prompt_usage_tokens, first_token_time, has_first_token_been_tracked = await parse_streaming_response(
                response.content, tokenizer, args
            )
        
        # Step 4: Calculate performance metrics
        if token_count > 0:
            # Get end time from the last response processing
            end_time = time.perf_counter()
            result.update(calculate_performance_metrics(
                start_time, first_token_time, end_time, token_count,
                prompt_usage_tokens, expected_pp_tokens, latency, args
            ))

    except Exception as e:
        print(f"Error during run: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 5: Execute post-run command
    execute_post_run_command(post_run_cmd)

    return result


async def main_async():
    args = parse_arguments()
    
    if args.enable_prefix_caching and args.no_cache:
        print("Error: --enable-prefix-caching and --no-cache are incompatible.")
        return

    version_number = __version__

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"llama-benchy ({version_number})")
    print(f"Date: {current_time}")
    print(f"Benchmarking model: {args.model} at {args.base_url}")
    
    served_model_name = args.served_model_name if args.served_model_name else args.model

    tokenizer = load_tokenizer(
        hf_name=args.tokenizer_hf,
        local_path=args.tokenizer_path
    )
    all_tokens = prepare_text_data(args.book_url, tokenizer)
    print(f"Total tokens available in text corpus: {len(all_tokens)}")
    
    # Use a large timeout for long-running benchmarks
    timeout = aiohttp.ClientTimeout(total=3600)
    connector = aiohttp.TCPConnector(limit=1, force_close=False, keepalive_timeout=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
        delta_user = 0
        delta_context = 0
        should_warmup = not args.no_warmup
        if args.adapt_prompt:
            should_warmup = True
            
        if should_warmup:
            delta_user, delta_context = await warmup(session, args.base_url, args.api_key, served_model_name, tokenizer if args.adapt_prompt else None)

        latency = await measure_latency(session, args.base_url, args.api_key, args.latency_mode, served_model_name)
        
        results = []
        
        for depth in args.depth:
            for pp in args.pp:
                for tg in args.tg:
                    print(f"Running test: pp={pp}, tg={tg}, depth={depth}")
                    pp_speeds = []
                    tg_speeds = []
                    ttft_values = []
                    ttfr_values = []
                    est_ppt_values = []
                    e2e_ttft_values = []
                    
                    ctx_pp_speeds = []
                    ctx_tg_speeds = []
                    ctx_ttfr_values = []
                    ctx_est_ppt_values = []
                    ctx_e2e_ttft_values = []
                    
                    for run in range(args.runs):
                        current_pp = pp
                        current_depth = depth
                        if args.adapt_prompt:
                            if depth == 0:
                                current_pp = max(1, pp - delta_user)
                            else:
                                current_depth = max(1, depth - delta_context)

                        context, prompt = generate_prompt(all_tokens, tokenizer, current_pp, current_depth, args.no_cache)
                        
                        if args.enable_prefix_caching and depth > 0:
                            # Request 1: Context only
                            # We send context as system message, and empty prompt as user message.
                            # This establishes the prefix: [System: Context] [User: ""]
                            # Expected PP tokens = current_depth (context size)
                            print(f"  Run {run+1}/{args.runs} (Context Load)...")
                            ctx_result = await run_benchmark(session, args.base_url, args.api_key, served_model_name, context, "", current_depth, tg, args.no_cache, latency, None, tokenizer, args)
                            
                            if ctx_result:
                                if ctx_result["pp_speed"] is not None:
                                    ctx_pp_speeds.append(ctx_result["pp_speed"])
                                if ctx_result["tg_speed"] is not None:
                                    ctx_tg_speeds.append(ctx_result["tg_speed"])
                                if ctx_result["ttfr"] is not None:
                                    ctx_ttfr_values.append(ctx_result["ttfr"])
                                if ctx_result["est_ppt"] is not None:
                                    ctx_est_ppt_values.append(ctx_result["est_ppt"])
                                if ctx_result["e2e_ttft"] is not None:
                                    ctx_e2e_ttft_values.append(ctx_result["e2e_ttft"])
                            
                            # Request 2: Context + Prompt
                            # We send context as system message, and prompt as user message.
                            # The prefix [System: Context] should be cached.
                            # Expected PP tokens = current_pp (prompt size only)
                            print(f"  Run {run+1}/{args.runs} (Inference)...")
                            run_result = await run_benchmark(session, args.base_url, args.api_key, served_model_name, context, prompt, current_pp, tg, args.no_cache, latency, args.post_run_cmd, tokenizer, args)
                        else:
                            # Standard run
                            # Expected PP tokens = current_pp + current_depth
                            expected_tokens = current_pp + current_depth
                            run_result = await run_benchmark(session, args.base_url, args.api_key, served_model_name, context, prompt, expected_tokens, tg, args.no_cache, latency, args.post_run_cmd, tokenizer, args)
                        
                        if run_result:
                            if run_result["tg_speed"] is not None:
                                tg_speeds.append(run_result["tg_speed"])
                            if run_result["pp_speed"] is not None:
                                pp_speeds.append(run_result["pp_speed"])
                            if run_result["est_ppt"] is not None:
                                est_ppt_values.append(run_result["est_ppt"])
                            if run_result["ttfr"] is not None:
                                ttfr_values.append(run_result["ttfr"])
                            if run_result["ttft"] is not None:
                                ttft_values.append(run_result["ttft"])
                            if run_result["e2e_ttft"] is not None:
                                e2e_ttft_values.append(run_result["e2e_ttft"])

                    # Aggregate results
                    def format_result(values, multiplier=1.0):
                        if not values: return ""
                        mean = np.mean(values) * multiplier
                        std = np.std(values) * multiplier
                        return f"{mean:.2f} ± {std:.2f}"

                    # Context PP (if enabled)
                    if ctx_pp_speeds:
                        test_name = f"ctx_pp @ d{depth}"
                        results.append([
                            args.model, 
                            test_name, 
                            format_result(ctx_pp_speeds), 
                            format_result(ctx_ttfr_values, 1000), 
                            format_result(ctx_est_ppt_values, 1000), 
                            format_result(ctx_e2e_ttft_values, 1000)
                        ])

                    # Context TG (if enabled)
                    if ctx_tg_speeds:
                        test_name = f"ctx_tg @ d{depth}"
                        results.append([args.model, test_name, format_result(ctx_tg_speeds), "", "", ""])

                    # Standard PP
                    if pp_speeds:
                        test_name = f"pp{pp}"
                        if depth > 0: test_name += f" @ d{depth}"
                        results.append([
                            args.model, 
                            test_name, 
                            format_result(pp_speeds), 
                            format_result(ttfr_values, 1000), 
                            format_result(est_ppt_values, 1000), 
                            format_result(e2e_ttft_values, 1000)
                        ])
                    
                    # Standard TG
                    if tg_speeds:
                        test_name = f"tg{tg}"
                        if depth > 0: test_name += f" @ d{depth}"
                        results.append([args.model, test_name, format_result(tg_speeds), "", "", ""])

        print()
        if not results:
            print("No results collected. Check if the model is generating tokens.")
        else:
            print(tabulate(results, headers=["model", "test", "t/s", "ttfr (ms)", "est_ppt (ms)", "e2e_ttft (ms)"], tablefmt="pipe", colalign=("left", "right", "right", "right", "right", "right")))
            print(f"\nllama-benchy ({version_number})")
            print(f"date: {current_time} | latency mode: {args.latency_mode}")


def main():
    """Entry point for the CLI command."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
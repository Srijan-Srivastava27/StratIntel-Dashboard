# Standard library imports for timing and function wrapping
import time
import openai
import functools

def retry_openai(func, max_retries=5, initial_delay=1):
    """
    Decorator to handle OpenAI API retries with exponential backoff
    
    Args:
        func: Function to wrap with retry logic
        max_retries: Maximum number of retry attempts (default: 5)
        initial_delay: Starting delay in seconds (default: 1)
    """
    @functools.wraps(func)  # Preserve original function metadata
    def wrapper(*args, **kwargs):
        # Initialize retry delay
        delay = initial_delay
        
        # Attempt API call with retries
        for attempt in range(max_retries):
            try:
                # Execute wrapped function
                return func(*args, **kwargs)
            except openai.error.RateLimitError:
                # Handle rate limiting with exponential backoff
                print(f"Rate limit reached. Retry in {delay:.1f} seconds.")
                time.sleep(1)
                delay *= 2  # Double delay for next attempt
            except openai.error.OpenAIError as e:
                # Handle other OpenAI-specific errors
                print(f"OpenAI API error: {e}, retry in {delay:.1f}s")
                time.sleep(1)
                delay *= 2  # Double delay for next attempt
        # If all retries failed, raise exception
        raise Exception("Failed after max retries to call OpenAI API.")
    return wrapper

@retry_openai  # Apply retry decorator
def call_openai_with_retry(**kwargs):
    """
    Make OpenAI API calls with automatic retry handling
    
    Args:
        **kwargs: Keyword arguments to pass to openai.Completion.create()
    
    Returns:
        OpenAI API response
        
    Raises:
        Exception: If all retry attempts fail
    """
    return openai.Completion.create(**kwargs)

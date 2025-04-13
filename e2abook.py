# ... (Keep imports and configuration the same, including EDGE_TTS_VOICE) ...
import asyncio
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import ebooklib
import edge_tts # Keep this
# Attempt to import the specific exception, fallback planned
try:
    from edge_tts import NoAudioReceived
    NO_AUDIO_EXCEPTION_TYPE = NoAudioReceived
except ImportError:
    print("Warning: Could not import edge_tts.NoAudioReceived. Using general Exception for this case.")
    NO_AUDIO_EXCEPTION_TYPE = None # Will check error message string instead

from bs4 import BeautifulSoup
from ebooklib import epub
from pydub import AudioSegment
from pypdf import PdfReader

# --- Configuration ---
EDGE_TTS_VOICE = "en-US-ChristopherNeural" # Or your preferred voice
OUTPUT_FORMAT = "mp3"
TEXT_CHUNK_MAX_CHARS = 1000
PAUSE_BETWEEN_CHUNKS_MS = 0

# --- NEW: Concurrency and Retry Settings ---
MAX_CONCURRENT_TASKS = 10 # Limit simultaneous edge-tts requests (Adjust as needed)
MAX_RETRIES = 2           # Number of retries for a failing chunk
RETRY_DELAY_SECONDS = 2   # Seconds to wait before retrying


# --- Text Extraction Functions (Unchanged - Keep extract_text_from_epub, txt, pdf) ---
# ... (Previous code for text extraction) ...
def extract_text_from_epub(filepath: Path) -> Optional[str]:
    """Extracts text content from an EPUB file using BeautifulSoup for cleaner parsing."""
    print(f"Extracting text from EPUB: {filepath.name}")
    try:
        book = epub.read_epub(filepath)
        full_text = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                content_bytes = item.get_content()
                try:
                    content_html = content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    content_html = content_bytes.decode('latin-1', errors='ignore')

                soup = BeautifulSoup(content_html, 'lxml')
                text_content = '\n\n'.join(p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']))
                if text_content:
                    full_text.append(text_content)
            except Exception as e:
                print(f"  Warning: Could not process item {item.get_name()}: {e}")
        print("EPUB extraction complete.")
        return "\n\n".join(full_text) if full_text else None
    except Exception as e:
        print(f"Error reading EPUB file {filepath.name}: {e}")
        return None

def extract_text_from_txt(filepath: Path) -> Optional[str]:
    """Extracts text content from a plain text file."""
    print(f"Extracting text from TXT: {filepath.name}")
    try:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
             with open(filepath, 'r', encoding='latin-1') as f:
                 return f.read()
    except Exception as e:
        print(f"Error reading TXT file {filepath.name}: {e}")
        return None

def extract_text_from_pdf(filepath: Path) -> Optional[str]:
    """Extracts text content from a PDF file. Quality depends heavily on PDF structure."""
    print(f"Extracting text from PDF: {filepath.name} (Quality may vary significantly)")
    full_text = []
    try:
        reader = PdfReader(filepath)
        print(f"  PDF has {len(reader.pages)} pages.")
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text.strip())
                else:
                    print(f"  Warning: No text extracted from page {i+1}.")
            except Exception as e:
                print(f"  Warning: Could not extract text from page {i+1}: {e}")
        print("PDF extraction complete.")
        return "\n\n".join(full_text) if full_text else None
    except Exception as e:
        print(f"Error reading PDF file {filepath.name}: {e}")
        return None


# --- Text Cleaning & Splitting (Unchanged - Keep clean_text, split_text_into_chunks) ---
# ... (Previous code for cleaning and splitting) ...
def clean_text(text: str) -> str:
    """Performs basic text cleaning."""
    if not text: return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text_into_chunks(text: str, max_chars: int = TEXT_CHUNK_MAX_CHARS) -> List[str]:
    """Splits text into manageable chunks for TTS, trying to respect sentences."""
    print(f"Splitting text into chunks (max ~{max_chars} chars)...")
    if not text: return []
    chunks = []
    # First, clean potentially problematic characters (optional, customize as needed)
    text = text.replace("\"", "'").replace("“", "'").replace("”", "'") # Example: Normalize quotes

    sentences = re.split(r'(?<=[.!?])\s+', text) # Basic sentence split
    current_chunk = ""
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean: continue

        if len(current_chunk) + len(sentence_clean) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence_clean + " "
        elif len(sentence_clean) > max_chars: # Handle single sentences that are too long
             if current_chunk: chunks.append(current_chunk.strip())
             # Split the long sentence itself
             for i in range(0, len(sentence_clean), max_chars):
                 chunks.append(sentence_clean[i:i+max_chars].strip())
             current_chunk = "" # Reset chunk
        else:
            current_chunk += sentence_clean + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # --- NEW: Filter out any empty strings just in case ---
    final_chunks = [chunk for chunk in chunks if chunk]
    print(f"Split into {len(final_chunks)} non-empty chunks.")
    return final_chunks


# --- 3. Text-to-Speech (edge-tts - Async with Retries) ---

async def text_to_speech_edge_with_retries(
    text_chunk: str,
    output_mp3_path: Path,
    voice: str = EDGE_TTS_VOICE,
    retries: int = MAX_RETRIES,
    delay: int = RETRY_DELAY_SECONDS
) -> bool:
    """Generates audio using edge-tts, with retries on failure."""
    attempts = 0
    while attempts <= retries:
        attempts += 1
        try:
            communicate = edge_tts.Communicate(text_chunk, voice)
            await communicate.save(str(output_mp3_path))
            # Check if file is valid
            if output_mp3_path.is_file() and output_mp3_path.stat().st_size > 100:
                if attempts > 1:
                     print(f"  Success after {attempts} attempts for chunk starting: '{text_chunk[:50]}...'")
                return True # Success!
            else:
                # File not created or too small, counts as failure for retry
                 if output_mp3_path.is_file(): output_mp3_path.unlink(missing_ok=True) # Clean up bad file
                 raise ValueError("Generated file is missing or too small.")

        except Exception as e:
            # --- Corrected Exception Handling ---
            is_no_audio_error = False
            if NO_AUDIO_EXCEPTION_TYPE and isinstance(e, NO_AUDIO_EXCEPTION_TYPE):
                 is_no_audio_error = True
                 error_type = "NoAudioReceived"
            # Fallback check if specific import failed or for other errors containing the text
            elif "NoAudioReceived" in str(e) or "no audio was received" in str(e).lower():
                 is_no_audio_error = True
                 error_type = "NoAudioReceived (detected by string)"
            else:
                 error_type = type(e).__name__

            print(f"  Attempt {attempts}/{retries+1} failed for chunk: '{text_chunk[:50]}...'. Error: {error_type}: {e}")

            # --- Log the failed text chunk for debugging ---
            # Consider writing failed chunks to a separate log file
            # print(f"    Failed Text: {text_chunk}")

            if attempts > retries:
                print(f"  Max retries reached. Giving up on this chunk.")
                return False # Failed after all retries

            print(f"  Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

    return False # Should not be reached, but safety return


# --- 4. Main Orchestration (Adapted for Async TTS with Semaphore) ---

async def generate_all_chunks_async(
    chunks_to_process: List[str],
    temp_dir: Path,
    semaphore: asyncio.Semaphore # <-- Pass semaphore here
) -> List[Path]:
    """Processes all text chunks asynchronously using edge-tts with concurrency limit."""
    print(f"\n--- Starting TTS Synthesis (using edge-tts) into {temp_dir} ---")
    tasks = []
    output_files_map = {} # Map task index to output file path

    async def process_one_chunk(index: int, chunk: str, output_path: Path):
        """Worker coroutine to process one chunk with semaphore."""
        async with semaphore: # Acquire semaphore before starting task
             print(f"    Starting TTS for chunk {index+1}/{len(chunks_to_process)}...")
             success = await text_to_speech_edge_with_retries(chunk, output_path)
             if success:
                 print(f"    Finished TTS for chunk {index+1}/{len(chunks_to_process)} -> {output_path.name}")
                 return output_path # Return path on success
             else:
                 print(f"    Failed TTS for chunk {index+1}/{len(chunks_to_process)}.")
                 # Optionally log failed text here or in the retry function
                 # with open("failed_chunks.log", "a", encoding="utf-8") as log_f:
                 #    log_f.write(f"Chunk {index+1}:\n{chunk}\n---\n")
                 return None # Indicate failure

    for i, chunk in enumerate(chunks_to_process):
        # Skip empty chunks just in case (already filtered, but double check)
        if not chunk:
            print(f"  Skipping empty chunk index {i}.")
            continue
        chunk_filename = temp_dir / f"chunk_{i:05d}.mp3"
        output_files_map[i] = chunk_filename
        # Create task using the worker coroutine
        tasks.append(
            asyncio.create_task(process_one_chunk(i, chunk, chunk_filename), name=f"Chunk_{i}")
        )

    # Wait for all tasks managed by the semaphore to complete
    print(f"\n  Waiting for TTS generation to complete (Concurrency: {MAX_CONCURRENT_TASKS})...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_files_dict = {} # Use dict to preserve order later
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Error processing Task for chunk index {i}: {result}")
            # Clean up potential partial file if it exists
            if i in output_files_map and output_files_map[i].exists():
                output_files_map[i].unlink(missing_ok=True)
        elif result is not None and isinstance(result, Path) and result.exists():
            # Result is the successful output path from process_one_chunk
             successful_files_dict[i] = result # Store path with original index
        # else: result was None (failure reported by process_one_chunk) - already logged

    # Sort successful files based on original chunk index
    successful_files = [successful_files_dict[key] for key in sorted(successful_files_dict.keys())]

    print(f"\n--- TTS Synthesis Complete ({len(successful_files)}/{len(chunks_to_process)} chunks successful) ---")
    return successful_files


def create_audiobook(ebook_path_str: str, output_path_str: str):
    """Main function to convert an ebook file to an audiobook using edge-tts."""
    start_time = time.time()
    ebook_path = Path(ebook_path_str)
    output_path = Path(output_path_str)

    # --- Input Validation (Unchanged) ---
    if not ebook_path.is_file(): # ... (rest of validation) ...
        print(f"Error: Input ebook file not found at '{ebook_path}'")
        return

    # --- Text Extraction (Unchanged) ---
    ext = ebook_path.suffix.lower()
    raw_text: Optional[str] = None
    if ext == '.epub': raw_text = extract_text_from_epub(ebook_path)
    elif ext == '.txt': raw_text = extract_text_from_txt(ebook_path)
    elif ext == '.pdf': raw_text = extract_text_from_pdf(ebook_path)
    else:
        print(f"Error: Unsupported file type '{ext}'.")
        return
    if not raw_text: # ... (rest of extraction checks) ...
        print("Error: No text could be extracted from the input file.")
        return

    # --- Text Processing (Calls updated chunking) ---
    text_chunks = split_text_into_chunks(raw_text, max_chars=TEXT_CHUNK_MAX_CHARS)
    if not text_chunks: # ... (rest of chunk checks) ...
        print("Error: Text extracted, but splitting resulted in no processable chunks.")
        return

    # --- Execute Async TTS and Audio Assembly ---
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # --- NEW: Create Semaphore ---
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

        # Run the asynchronous part using the updated function
        successful_chunk_files = asyncio.run(
            generate_all_chunks_async(text_chunks, temp_dir_path, semaphore)
        )

        if not successful_chunk_files:
            print("Error: No audio chunks were successfully generated.")
            return

        # --- Audio Assembly (Mostly Unchanged, ensure using MP3) ---
        print("\n--- Assembling Audio Chunks ---")
        combined_audio = AudioSegment.empty()
        pause = AudioSegment.silent(duration=PAUSE_BETWEEN_CHUNKS_MS)

        # successful_chunk_files list is already sorted by original index
        for i, chunk_file in enumerate(successful_chunk_files):
            try:
                segment = AudioSegment.from_mp3(chunk_file) # Load MP3
                combined_audio += segment
                if PAUSE_BETWEEN_CHUNKS_MS > 0 and i < len(successful_chunk_files) - 1:
                    combined_audio += pause
            except Exception as e:
                print(f"  Warning: Could not load or append chunk {chunk_file.name}: {e}")
                # Consider logging which file failed to load

        if len(combined_audio) == 0:
            print("Error: Failed to assemble any audio from the generated chunks.")
            return

        # --- Export Final Audio (Unchanged) ---
        print(f"\nExporting final audio to '{output_path}' (Format: {OUTPUT_FORMAT})")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_audio.export(output_path, format=OUTPUT_FORMAT, bitrate='192k')
            end_time = time.time()
            print("-" * 30)
            print(f"Audiobook creation complete! 🎉")
            print(f"Output file: {output_path}")
            print(f"Total processing time: {end_time - start_time:.2f} seconds")
            print(f"Successful Chunks: {len(successful_chunk_files)} / {len(text_chunks)}")
            print("-" * 30)
        except Exception as e:
            print(f"Error exporting final audio file: {e}")
            print("Make sure ffmpeg is installed and accessible.")

    # Temp directory cleaned up automatically

# --- Main Execution (Unchanged) ---
if __name__ == "__main__":
    INPUT_EBOOK_FILE = ".epub" # Use raw string or forward slashes or escape your escapes
    OUTPUT_AUDIOBOOK_FILE = "audio.mp3" # Use raw string or forward slashes or escape your escapes

    create_audiobook(INPUT_EBOOK_FILE, OUTPUT_AUDIOBOOK_FILE)
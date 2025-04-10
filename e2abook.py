import asyncio
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import ebooklib
import edge_tts
from bs4 import BeautifulSoup
from ebooklib import epub
from pydub import AudioSegment
from pypdf import PdfReader

# --- Configuration ---
# Select an edge-tts voice. Find more voices: edge-tts --list-voices
# Example voices: en-US-AriaNeural, en-US-JennyNeural, en-GB-SoniaNeural, etc.
EDGE_TTS_VOICE = "en-US-EmmaNeural"

OUTPUT_FORMAT = "mp3"
# Edge-TTS can sometimes handle slightly longer chunks, but keep reasonable
TEXT_CHUNK_MAX_CHARS = 500
PAUSE_BETWEEN_CHUNKS_MS = 150 # Pause between combined audio chunks

# --- Helper: Ensure ffmpeg is available (Same as before) ---
# Basic check, pydub might find it automatically if in PATH


# --- 1. Text Extraction Functions (Mostly Unchanged) ---

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


# --- 2. Text Cleaning & Splitting (Unchanged) ---

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
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = ""
    for sentence in sentences:
        if not sentence: continue
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
        elif len(sentence) > max_chars:
             if current_chunk: chunks.append(current_chunk.strip())
             for i in range(0, len(sentence), max_chars):
                 chunks.append(sentence[i:i+max_chars].strip())
             current_chunk = ""
        else:
            current_chunk += sentence + " "
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    print(f"Split into {len(chunks)} chunks.")
    return chunks


# --- 3. Text-to-Speech (edge-tts - Async) ---

async def text_to_speech_edge(
    text_chunk: str,
    output_mp3_path: Path,
    voice: str = EDGE_TTS_VOICE
) -> bool:
    """Generates audio from a text chunk using edge-tts and saves directly to MP3."""
    try:
        communicate = edge_tts.Communicate(text_chunk, voice)
        await communicate.save(str(output_mp3_path)) # communicate.save needs string path
        # Check if file was created and has size > 0 (basic check)
        if output_mp3_path.is_file() and output_mp3_path.stat().st_size > 100:
             return True
        else:
             print(f"  Warning: edge-tts completed but output file seems invalid for chunk: '{text_chunk[:50]}...'")
             # Attempt to remove potentially empty/corrupt file
             if output_mp3_path.is_file(): output_mp3_path.unlink()
             return False
    except edge_tts.NoAudioReceived:
         print(f"  Error: edge-tts received no audio for chunk: '{text_chunk[:50]}...'")
         return False
    except Exception as e:
        print(f"  Error synthesizing chunk with edge-tts: {e} - Chunk: '{text_chunk[:50]}...'")
        return False


# --- 4. Main Orchestration (Adapted for Async TTS) ---

def create_audiobook(ebook_path_str: str, output_path_str: str):
    """Main function to convert an ebook file to an audiobook using edge-tts."""
    start_time = time.time()
    ebook_path = Path(ebook_path_str)
    output_path = Path(output_path_str)

    # --- Input Validation ---
    if not ebook_path.is_file():
        print(f"Error: Input ebook file not found at '{ebook_path}'")
        return

    # --- Text Extraction ---
    ext = ebook_path.suffix.lower()
    raw_text: Optional[str] = None
    if ext == '.epub':
        raw_text = extract_text_from_epub(ebook_path)
    elif ext == '.txt':
        raw_text = extract_text_from_txt(ebook_path)
    elif ext == '.pdf':
        raw_text = extract_text_from_pdf(ebook_path)
    else:
        print(f"Error: Unsupported file type '{ext}'. Please use .epub, .txt, or .pdf.")
        return

    if not raw_text:
        print("Error: No text could be extracted from the input file.")
        return

    # --- Text Processing ---
    text_chunks = split_text_into_chunks(raw_text, max_chars=TEXT_CHUNK_MAX_CHARS)
    if not text_chunks:
        print("Error: Text extracted, but splitting resulted in no processable chunks.")
        return

    # --- Async TTS Generation ---
    async def generate_all_chunks_async(chunks_to_process: List[str], temp_dir: Path) -> List[Path]:
        """Processes all text chunks asynchronously using edge-tts."""
        print(f"\n--- Starting TTS Synthesis (using edge-tts) into {temp_dir} ---")
        tasks = []
        output_files = [] # Keep track of expected output paths
        for i, chunk in enumerate(chunks_to_process):
            if not chunk: continue
            chunk_filename = temp_dir / f"chunk_{i:05d}.mp3" # Edge-tts saves as mp3
            output_files.append(chunk_filename)
            # Create an async task for each chunk
            tasks.append(
                asyncio.create_task(text_to_speech_edge(chunk, chunk_filename, EDGE_TTS_VOICE), name=f"Chunk_{i}")
            )
            print(f"  Scheduled chunk {i+1}/{len(chunks_to_process)}...")

        # Wait for all tasks to complete, get results (True/False) or exceptions
        print("\n  Waiting for TTS generation to complete (may take time)...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_files: List[Path] = []
        for i, result in enumerate(results):
             # Check if the corresponding file path exists (double check)
             expected_file = output_files[i]
             if isinstance(result, Exception):
                 print(f"  Error in Task {i+1}: {result}")
                 # Attempt cleanup of potentially partially created file
                 if expected_file.exists(): expected_file.unlink(missing_ok=True)
             elif result is True and expected_file.exists(): # Check result AND file existence
                 successful_files.append(expected_file)
             else: # result was False or file doesn't exist
                 print(f"  Task {i+1} reported failure or file missing.")
                 if expected_file.exists(): expected_file.unlink(missing_ok=True) # Cleanup

        print(f"\n--- TTS Synthesis Complete ({len(successful_files)}/{len(chunks_to_process)} chunks successful) ---")
        return successful_files # Return list of successfully created MP3 chunk files

    # --- Execute Async TTS and Audio Assembly ---
    # Use a temporary directory for intermediate MP3 files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Run the asynchronous part of the process
        successful_chunk_files = asyncio.run(generate_all_chunks_async(text_chunks, temp_dir_path))

        if not successful_chunk_files:
            print("Error: No audio chunks were successfully generated.")
            return # Temp directory cleaned up automatically by 'with' statement

        print("\n--- Assembling Audio Chunks ---")
        # Combine chunks using pydub - important: files are now MP3
        combined_audio = AudioSegment.empty()
        pause = AudioSegment.silent(duration=PAUSE_BETWEEN_CHUNKS_MS)

        # Sort files numerically just in case asyncio order wasn't perfect (it should be)
        successful_chunk_files.sort()

        for i, chunk_file in enumerate(successful_chunk_files):
            try:
                # Load the mp3 chunk
                segment = AudioSegment.from_mp3(chunk_file)
                combined_audio += segment
                if PAUSE_BETWEEN_CHUNKS_MS > 0 and i < len(successful_chunk_files) - 1:
                    combined_audio += pause
            except Exception as e:
                print(f"  Warning: Could not load or append chunk {chunk_file.name}: {e}")

        if len(combined_audio) == 0:
             print("Error: Failed to assemble any audio from the generated chunks.")
             return

        # --- Export Final Audio ---
        print(f"\nExporting final audio to '{output_path}' (Format: {OUTPUT_FORMAT})")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_audio.export(output_path, format=OUTPUT_FORMAT, bitrate='192k') # Specify bitrate
            end_time = time.time()
            print("-" * 30)
            print(f"Audiobook creation complete! 🎉")
            print(f"Output file: {output_path}")
            print(f"Total processing time: {end_time - start_time:.2f} seconds")
            print("-" * 30)
        except Exception as e:
            print(f"Error exporting final audio file: {e}")
            print("Make sure ffmpeg is installed and accessible.")

    # Temporary directory is automatically cleaned up here


# --- Main Execution ---
if __name__ == "__main__":
   # Dont forget to escape your backslashes for windows paths or use raw strings
    # --- USER: Set your input and output file paths here ---
    INPUT_EBOOK_FILE = ".epub"  # Example: "C:/Users/You/Downloads/my_book.epub"
    OUTPUT_AUDIOBOOK_FILE = ".mp3" # Example: "C:/Users/You/Music/my_audiobook.mp3"

    # You can change the voice in the Configuration section at the top
    create_audiobook(INPUT_EBOOK_FILE, OUTPUT_AUDIOBOOK_FILE)

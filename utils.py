from google import genai
from PIL import Image
import pillow_avif
import re
from pathlib import Path
import requests
import os
import mimetypes
from urllib.parse import urlparse
from dotenv import load_dotenv
from pydantic import BaseModel
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MAX_TOKENS_PER_CHUNK = 4096
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-clip-v2")
CHUNK_OVERLAP = 400


class ImageDescription(BaseModel):
    file_name: str
    description: str


def download_image(url, filename, save_dir="images", with_ext=False):
    """
    Downloads an image from a URL and saves it with the correct extension.

    Args:
        url (str): The URL of the image to download.
        save_dir (str): The directory to save the image in.
        filename (str): Name of the file
    """
    try:
        # --- 1. Make the request and check for errors ---
        # Using stream=True is efficient for downloading large files
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # --- 2. Determine the file extension from the Content-Type header ---
        content_type = response.headers.get("content-type")
        if not content_type:
            print(f"Warning: Could not determine content type for {url}.")
            return

        # Use the mimetypes library to guess the extension
        extension = mimetypes.guess_extension(content_type)
        if not extension:
            # A fallback for unknown image types, or types mimetypes doesn't know
            # e.g., image/webp might not be in older Python's mimetypes db
            if "jpeg" in content_type:
                extension = ".jpg"
            elif "png" in content_type:
                extension = ".png"
            else:
                print(
                    f"Warning: Could not determine file extension for MIME type {content_type}."
                )
                return

        # --- 3. Determine the filename ---
        # Parse the URL to get the path, then get the last part of the path
        parsed_url = urlparse(url)
        # Get the filename and remove any extension it might already have
        filename_from_url = os.path.basename(parsed_url.path)
        filename_base = os.path.splitext(filename_from_url)[0]

        # If the filename is empty (e.g., from a URL like 'https://site.com/'), create a generic one
        if not filename_base:
            filename_base = "image"

        filename = (filename + extension) if not with_ext else filename

        # --- 4. Create the directory and save the file ---
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        # Write the content to the file in chunks
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded and saved '{filename}' to '{save_dir}/'")
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")
        return None


# --- CONFIGURE THE API KEY ---
# It's best practice to set this as an environment variable, but for simplicity:
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("DISCOURSE_URL")
client = genai.Client(api_key=GOOGLE_API_KEY)


def describe_image_with_gemini(path_to_images: list[str], context_prompt: str) -> str:
    """
        Uses the Gemini 1.5 Flash model to describe an image from a local path.

        Args:
            image_path (str): The local file path to the image.
        context_prompt (str): The text prompt providing context for the image.

    Returns:
        str: A detailed text description of the image, or an error message.
    """
    # try:
    # Load the image using Pillow
    imgs = [Image.open(x).convert("RGBA") for x in path_to_images]

    # Select the model. 'gemini-1.5-flash-latest' is perfect for "lite" and fast use.
    # It's highly capable and cost-effective.
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[context_prompt, *imgs],
        config={
            "response_mime_type": "application/json",
            "response_schema": list[ImageDescription],
        },
    )
    # print(response.text)

    # print(f"  [Gemini API] Describing '{os.path.basename(image_path)}'...")

    # The prompt is a list containing the text and the image
    # response = model.generate_content([context_prompt, img], generation_config={ 'response' })

    # You can add more safety checks here if needed
    # response.prompt_feedback

    return response

    # except FileNotFoundError:
    #     return f"[Error: Image file not found at in th paths]"
    # except Exception as e:
    #     # Catch other potential API errors (e.g., authentication, rate limits)
    #     print(f"  [Gemini API] An error occurred: {e}")
    #     return f"[Error: Could not get description from Gemini API. Details: {e}]"


# (Assume the describe_image_with_gemini function from above is in the same file)


def download_image_from_markdown(markdown_text, id, savedir):
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

    matches = list(image_pattern.finditer(markdown_text))
    print(matches)

    if len(matches) == 0:
        print("No images to download")
        return

    print("Found", len(matches), "images")
    for i, match in enumerate(reversed(matches), start=1):
        alt_text = match.group(1)
        download_path = match.group(2)
        download_url = (
            BASE_URL + "/uploads/short-url/" + download_path.removeprefix("upload://")
        )
        print(f"Downloading image {i} of {len(matches)}")
        download_image(
            download_url,
            str(id) + "_" + download_path.removeprefix("upload://"),
            savedir,
            with_ext=True,
        )
        print("Downloaded successfully")


def embed_image_descriptions(chunk: str, descriptions: dict[str, str]):
    """
    Finds all image tags in markdown, generates descriptions using Gemini,
    and rewrites the text.
    """
    image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

    matches = list(image_pattern.finditer(chunk))

    for match in reversed(matches):
        image_path = match.group(2)
        # print(image_path)
        image_description = descriptions.get(
            image_path.removeprefix("upload://").split(".")[0], "None"
        )

        enriched_tag = f"\n[Image Description: {image_description.strip()}]\n"

        start, end = match.span()
        chunk = chunk[:start] + enriched_tag + chunk[end:]

    return chunk


def build_reply_hierarchy(flat_posts: list[dict]) -> list[dict]:
    """
    Builds a nested reply hierarchy from a flat list of post objects.

    Each post object in the input list must have at least 'id' and 'reply_to_post_id'.
    'reply_to_post_id' should be None for top-level posts.

    Args:
        flat_posts: A list of dictionaries, where each dictionary represents a post.

    Returns:
        A list of top-level post dictionaries, with child posts nested
        under a 'replies' key.
    """
    post_map = {}

    # --- First Pass: Map all posts by their ID and initialize 'replies' list ---
    for post in flat_posts:
        post_id = post["post_number"]
        post["replies"] = []  # Add the replies list to each post object
        post_map[post_id] = post

    hierarchy = []

    # --- Second Pass: Connect replies to their parents ---
    for post in flat_posts:
        parent_id = post.get("reply_to_post_number")  # Use .get() for safety

        if parent_id is not None:
            # This is a reply to another post
            if parent_id in post_map:
                # Find the parent in our map and append this post as a reply
                parent_post = post_map[parent_id]
                parent_post["replies"].append(post)
            else:
                # This is an orphan post, its parent doesn't exist in the list.
                # We can choose to ignore it, or add it as a top-level post.
                # Here, we'll treat it as a top-level post for robustness.
                print(
                    f"Warning: Post {post['post_number']} has a missing parent {parent_id}. Treating as top-level."
                )
                hierarchy.append(post)
        else:
            # This is a top-level post (a new thread)
            hierarchy.append(post)

    return hierarchy


def create_contextual_chunks(nested_threads: list[dict]) -> list[dict]:
    """
    Traverses the nested thread structure to create context-rich text chunks
    for RAG embedding.

    Args:
        nested_threads: The output from the build_reply_hierarchy function.
        topic_title: The title of the parent topic for top-level context.

    Returns:
        A flat list of dictionaries, each ready for embedding, containing the
        original post data and the new 'context_chunk'.
    """
    all_chunks = []

    def process_node(node: dict, parent_context: str, depth: int):
        # Create a prefix for visual indentation in the chunk
        prefix = ">" * depth
        if prefix:
            prefix += " "

        # The current post's text, prefixed to show it's part of a thread
        current_post_text = f"{prefix}Post {node['id']}: {node['raw']}"

        # Combine parent context with the current post's text
        context_chunk = f"{parent_context}\n{current_post_text}".strip()

        # Add the final chunk to our list
        all_chunks.append(
            {
                "topic_id": node["topic_id"],
                "post_id": node["id"],
                "content": context_chunk,
                "original_text": node["raw"],
                "url": node["post_url"],
            }
        )

        # Recurse for all replies
        for reply in node["replies"]:
            # The new context for the children is the full context we just built
            process_node(reply, parent_context=context_chunk, depth=depth + 1)

    # Start the process for each top-level thread
    for thread_start_node in nested_threads:
        # Top-level posts get the main Topic as their context
        # top_level_context = f"Topic: {topic_title}"
        process_node(
            thread_start_node,
            parent_context="Topic: " + thread_start_node["title"] + "\n",
            depth=0,
        )

    return all_chunks


def create_hierarchical_chunks(nested_threads: list[dict]) -> list[dict]:
    """
    The master function to create optimized, hierarchical chunks for a large-context RAG system.

    It traverses a nested conversation tree and for each post, it:
    1.  Constructs a context string by including parent posts, prioritizing the most recent.
    2.  Respects the MAX_TOKENS_PER_CHUNK limit for the combined hierarchical context.
    3.  If a single post's text *itself* exceeds the limit, it splits that post
        into linked sub-chunks.

    Returns:
        A flat list of dictionaries, each a chunk ready for embedding and upserting.
    """
    all_final_chunks = []

    def process_node(node: dict, parent_history: list[str]):
        # parent_history is a list of formatted parent strings, e.g., ["Post 101: ...", "> Post 102: ..."]

        current_post_text = node["raw"]
        current_post_id = node["id"]
        prefix = ">" * len(parent_history) + " " if parent_history else ""
        formatted_current_post = f"{prefix}Post {current_post_id}: {current_post_text}"

        # --- Token Budgeting for Hierarchy ---
        context_parts = []
        token_budget = MAX_TOKENS_PER_CHUNK

        # 1. Add current post's text to the budget first
        # We must check if the current post *alone* is too big
        current_post_tokens = len(tokenizer.encode(formatted_current_post))

        if current_post_tokens > MAX_TOKENS_PER_CHUNK:
            # --- STRATEGY A: HANDLE MONOLITH POST (Self-Splitting) ---
            print(
                f"INFO: Post {current_post_id} is a monolith ({current_post_tokens} tokens). Splitting it."
            )

            # Create a header with the available parent context that fits
            parent_context_header = build_parent_context(
                parent_history, token_budget // 4
            )  # Use 1/4 budget for parent context
            full_header = f"Topic: {node['title']}\n{parent_context_header}\n"

            # Split the monolith post's raw text
            split_post_texts = split_text_only(current_post_text)
            total_splits = len(split_post_texts)

            for i, part in enumerate(split_post_texts):
                chunk_header = f"{full_header}Post: {current_post_id} (Part {i + 1}/{total_splits})\n\n"
                final_chunk_text = chunk_header + part

                all_final_chunks.append(
                    {
                        "chunked_id": f"{current_post_id}-chunk-{i}",
                        "topic_id": node["topic_id"],
                        "post_id": node["id"],
                        "topic_title": node["title"],
                        "chunk_index": i,
                        "total_chunks": total_splits,
                        "content": final_chunk_text,
                        "url": node["url"],
                    }
                )
            # After splitting, continue recursion with the monolith's *full text* as history for its children
            new_history_for_children = parent_history + [formatted_current_post]
            for reply in node.get("replies", []):
                process_node(reply, new_history_for_children)
            return  # Stop further processing for this monolith node itself

        # --- STRATEGY B: HANDLE REGULAR POST (Context Packing) ---
        context_parts.append(formatted_current_post)
        token_budget -= current_post_tokens

        # 2. Add parent context (most recent first) until budget is full
        for parent_text in reversed(parent_history):
            parent_tokens = len(tokenizer.encode(parent_text))
            if token_budget - parent_tokens > 0:
                context_parts.append(parent_text)
                token_budget -= parent_tokens
            else:
                break

        # 3. Add Topic Title if it fits
        topic_header = f"Topic: {node['title']}"
        if token_budget - len(tokenizer.encode(topic_header)) > 0:
            context_parts.append(topic_header)

        # Build the final chunk text in correct chronological order
        final_chunk_text = "\n\n".join(reversed(context_parts))

        all_final_chunks.append(
            {
                "chunked_id": f"{current_post_id}-chunk-{0}",
                "topic_id": node["topic_id"],
                "post_id": node["id"],
                "topic_title": node["title"],
                "chunk_index": 0,
                "total_chunks": 1,
                "content": final_chunk_text,
                "url": node["url"],
            }
        )

        # --- Recurse for children ---
        new_history_for_children = parent_history + [formatted_current_post]
        for reply in node.get("replies", []):
            process_node(reply, new_history_for_children)

    # --- Helper sub-functions ---
    def build_parent_context(history: list[str], budget: int) -> str:
        parts = []
        for parent_text in reversed(history):
            if budget - len(tokenizer.encode(parent_text)) > 0:
                parts.append(parent_text)
            else:
                break
        return "\n".join(reversed(parts))

    def split_text_only(text: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_TOKENS_PER_CHUNK - 500,  # Leave room for headers
            chunk_overlap=CHUNK_OVERLAP,
            length_function=lambda txt: len(tokenizer.encode(txt)),
        )
        return text_splitter.split_text(text)

    # --- Start the process for each top-level thread ---
    for thread_start_node in nested_threads:
        process_node(thread_start_node, parent_history=[])

    return all_final_chunks

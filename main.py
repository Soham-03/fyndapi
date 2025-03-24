import os
import uuid
import json
import re
from typing import Dict, Optional, List

import requests
from urllib.parse import urlparse

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# PDF processing
from pypdf import PdfReader

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')
PINECONE_INDEX_NAME = 'fynd-ai-unified-content-index'

# Initialize models
genai.configure(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
try:
    # Try to describe the index to check if it exists
    existing_index = pc.describe_index(PINECONE_INDEX_NAME)
    print(f"Index {PINECONE_INDEX_NAME} already exists.")
except Exception as e:
    # If the index doesn't exist, create it
    pc.create_index(
        name=PINECONE_INDEX_NAME, 
        dimension=embedding_model.get_sentence_embedding_dimension(),
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created index {PINECONE_INDEX_NAME}")

# Get the index
index = pc.Index(PINECONE_INDEX_NAME)

app = FastAPI()

# URL Validation Utility
def validate_url(url: str, content_type: str) -> bool:
    """
    Validate URL based on content type
    """
    try:
        # Validate URL structure
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return False
        
        # Content type specific validation
        if content_type == 'image':
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
            if not any(url.lower().endswith(ext) for ext in allowed_extensions):
                return False
        elif content_type == 'pdf':
            if not url.lower().endswith('.pdf'):
                return False
        
        # Attempt to download and verify
        response = requests.head(url, timeout=5)
        return (
            response.status_code == 200 and 
            content_type in response.headers.get('Content-Type', '').lower()
        )
    except Exception:
        return False

# Download Utility
def download_file(url: str, file_id: str, content_type: str) -> str:
    """
    Download file from URL
    """
    try:
        # Download file
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save downloaded file
        file_extension = 'pdf' if content_type == 'pdf' else 'jpg'
        file_path = f"temp_{file_id}_downloaded.{file_extension}"
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        return file_path
    
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error downloading {content_type}: {str(e)}"
        )

# Content Extraction Utilities
def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF
    """
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        
        for page in reader.pages:
            full_text += page.extract_text() + "\n\n"
        
        return full_text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error extracting PDF text: {str(e)}"
        )

def generate_image_description(image_path: str) -> str:
    """
    Generate a detailed, context-rich image description
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Open the image
        with open(image_path, 'rb') as img_file:
            image_parts = [
                {
                    'mime_type': 'image/jpeg',
                    'data': img_file.read()
                }
            ]
        
        # Comprehensive prompt
        prompt = """Provide an extremely detailed description of this image. 
        Include:
        - Detailed context of the scene
        - People present (number, activity, posture, clothing)
        - Technology or environment details
        - Color palette and composition
        - Specific elements that capture the essence of the scene
        Write in a way that would help someone identify or search for this exact image."""
        
        response = model.generate_content(
            contents=[prompt, image_parts[0]]
        )
        
        return response.text
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating image description: {str(e)}"
        )

# Summary and Tag Generation
def generate_content_summary(text: str, content_type: str) -> str:
    """
    Generate comprehensive summary using Gemini
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Comprehensive summary prompt
        prompt = f"""Create an extremely detailed, search-optimized summary of this {content_type}. 
        The summary should:
        - Capture the core content and key points
        - Provide context and main themes
        - Include potential search keywords
        - Be comprehensive enough to help someone understand the document's essence
        
        Limit to 1000 words.
        
        Document Text: {text[:10000]}  # Limit input to first 10000 characters
        """
        
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating {content_type} summary: {str(e)}"
        )

def generate_content_tags(summary: str) -> List[str]:
    """
    Generate tags from content summary
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Tag generation prompt
        prompt = f"""Extract the most relevant and specific tags from this content summary. 
        Focus on creating tags that capture:
        - Main topics
        - Key themes
        - Specific domains
        - Potential search terms
        
        Provide 7-10 precise, descriptive tags.
        Summary: {summary}
        
        Return as a comma-separated list of tags."""
        
        response = model.generate_content(prompt)
        
        # Clean and process tags
        tags = [
            tag.strip().lower() 
            for tag in response.text.split(',') 
            if tag.strip()
        ]
        
        # Ensure we have tags
        return tags[:10] if tags else ['content']
    
    except Exception as e:
        print(f"Tag generation error: {e}")
        return ['content']

# Keyword Extraction
def extract_keywords_from_query(query: str) -> List[str]:
    """
    Extract meaningful keywords from the query
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Prompt to extract key search terms
        prompt = f"""From the following search query, extract the most important 
        keywords and concepts that capture the essence of what is being searched.
        Focus on nouns, descriptive terms, and key context.
        
        Query: {query}
        
        Return as a comma-separated list of keywords without any additional text."""
        
        response = model.generate_content(prompt)
        
        # Clean and process keywords
        keywords = [
            keyword.strip().lower() 
            for keyword in response.text.split(',') 
            if keyword.strip()
        ]
        
        return keywords[:7]  # Limit to 7 keywords
    
    except Exception as e:
        # Fallback to simple keyword extraction
        print(f"Keyword extraction error: {e}")
        
        # Basic keyword extraction
        stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'for', 'of', 'with', 'and'])
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words]

# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    content_type: Optional[str] = None
    top_k: int = 5

@app.post("/process-content/")
async def process_content(
    content_url: str = Form(...),
    content_type: str = Form(...),
    user_id: Optional[str] = Form(None)
):
    """
    Process and index content (image or PDF) from a URL
    """
    # Validate content URL
    if not validate_url(content_url, content_type):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid or inaccessible {content_type} URL"
        )
    
    # Generate unique ID for the content
    content_id = str(uuid.uuid4())
    
    try:
        # Download content
        file_path = download_file(content_url, content_id, content_type)
        
        # Extract content and generate summary
        if content_type == 'pdf':
            # Extract text for PDFs
            full_text = extract_pdf_text(file_path)
            summary = generate_content_summary(full_text, content_type)
        else:
            # Generate description for images
            full_text = generate_image_description(file_path)
            summary = full_text
        
        # Generate tags
        tags = generate_content_tags(summary)
        
        # Create embedding from summary
        embedding = embedding_model.encode(summary).tolist()
        
        # Prepare metadata
        content_metadata = {
            'id': content_id,
            'url': content_url,
            'type': content_type,
            'user_id': user_id,
            'tags': tags,
            'summary': summary,
            'full_text': full_text[:5000]  # Store first 5000 characters
        }
        
        # Upsert to Pinecone
        index.upsert(vectors=[(content_id, embedding, content_metadata)])
        
        # Clean up temporary file
        os.remove(file_path)
        
        return {
            "message": f"{content_type.upper()} processed successfully",
            "content_id": content_id,
            "summary": summary,
            "generated_tags": tags
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing {content_type}: {str(e)}"
        )

@app.post("/search-content/")
async def search_content(query_request: QueryRequest):
    """
    Advanced semantic content search with enhanced tag matching
    """
    # Extract keywords from query
    query_keywords = extract_keywords_from_query(query_request.query)
    
    # Create query embedding
    query_embedding = embedding_model.encode(query_request.query).tolist()
    
    # Prepare query parameters
    query_params = {
        'vector': query_embedding,
        'top_k': query_request.top_k * 20,  # Fetch many more results
        'include_metadata': True
    }
    
    # Add content type filter if specified
    if query_request.content_type:
        query_params['filter'] = {
            'type': {'$eq': query_request.content_type}
        }
    
    # Query Pinecone
    query_result = index.query(**query_params)
    
    # Process results
    processed_results = []
    for match in query_result['matches']:
        metadata = match['metadata']
        summary = metadata.get('summary', '').lower()
        tags = metadata.get('tags', [])
        full_text = metadata.get('full_text', '').lower()
        
        # Enhanced tag matching
        tag_matches = sum(
            1 for keyword in query_keywords 
            for tag in tags 
            if keyword in tag or tag in keyword
        )
        
        # Keyword matching across summary, tags, and full text
        keyword_matches = sum(
            1 for keyword in query_keywords 
            if any(keyword in text for text in [summary, full_text] + tags)
        )
        
        # Semantic similarity scoring
        semantic_score = match['score']
        
        # Compute a combined relevance score with emphasis on tags
        combined_score = (
            semantic_score * 0.5 +  # Vector similarity weight
            (keyword_matches / max(len(query_keywords), 1)) * 0.3 +  # Keyword relevance
            (tag_matches / max(len(query_keywords), 1)) * 0.2  # Tag matching weight
        )
        
        # Debugging print
        print(f"Match Debug:")
        print(f"ID: {match['id']}")
        print(f"Type: {metadata.get('type')}")
        print(f"Tags: {tags}")
        print(f"Semantic Score: {semantic_score}")
        print(f"Keyword Matches: {keyword_matches}")
        print(f"Tag Matches: {tag_matches}")
        print(f"Combined Score: {combined_score}")
        
        # More flexible filtering
        if combined_score > 0.3:  # Adjusted threshold
            result = {
                'id': match['id'],
                'type': metadata.get('type'),
                'url': metadata.get('url'),
                'tags': tags,
                'summary': summary,
                'semantic_score': semantic_score,
                'keyword_matches': keyword_matches,
                'tag_matches': tag_matches,
                'combined_relevance_score': combined_score
            }
            processed_results.append(result)
    
    # Sort results by combined relevance score
    processed_results.sort(
        key=lambda x: x['combined_relevance_score'], 
        reverse=True
    )
    
    return {
        "query": query_request.query,
        "query_keywords": query_keywords,
        "results": processed_results[:5]  # Top 5 results
    }

# Health check endpoint
@app.get("/health/")
async def health_check():
    """
    Simple health check endpoint
    """
    try:
        # Check Pinecone connection
        index.describe_index_stats()
        
        return {
            "status": "healthy",
            "services": {
                "pinecone": "connected",
                "gemini": "configured"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Service unavailable: {str(e)}"
        )

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler to provide more informative error responses
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
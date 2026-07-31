"""
🤖 AI LIBRARY ASSISTANT - Production Ready Version
A comprehensive book discovery platform with AI-powered summaries and recommendations.
Combines Open Library API with Hugging Face models for intelligent book insights.

Author: Your Name
Version: 2.0.0
"""

import streamlit as st
import os
import json
import requests
import random
from datetime import datetime
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="AI Library Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment variables - use st.secrets for Streamlit Cloud, fallback to os.getenv for local
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
OPEN_LIBRARY_API = "https://openlibrary.org"

# Constants
DEFAULT_AGE = 25
MAX_SEARCH_RESULTS = 10
SUMMARY_LENGTH = 250
STUDENT_DISCOUNT = 0.25  # 25% discount

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables."""
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'age': DEFAULT_AGE,
            'is_student': False,
            'profile_saved': False,
            'username': None
        }
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    
    if 'current_search_results' not in st.session_state:
        st.session_state.current_search_results = []
    
    if 'selected_book' not in st.session_state:
        st.session_state.selected_book = None
    
    if 'ai_summary' not in st.session_state:
        st.session_state.ai_summary = ""
    
    if 'show_favorites' not in st.session_state:
        st.session_state.show_favorites = False

init_session_state()

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_ai_model():
    """
    Load Hugging Face text generation model.
    Uses GPT-2 for fast, efficient text generation.
    Falls back gracefully if HF_TOKEN is unavailable.
    """
    if not HF_TOKEN or HF_TOKEN == "your_token_here":
        logger.warning("HF_TOKEN not properly configured. AI features disabled.")
        return None
    
    try:
        model = pipeline(
            "text-generation",
            model="gpt2",
            token=HF_TOKEN,
            max_length=300,
            device=-1  # Use CPU, set to 0 for GPU
        )
        logger.info("✅ AI model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load AI model: {str(e)}")
        return None

ai_model = load_ai_model()

# ============================================================================
# API FUNCTIONS - BOOK SEARCH & DATA FETCHING
# ============================================================================

def search_open_library(query: str, limit: int = MAX_SEARCH_RESULTS) -> list:
    """
    Search for books using Open Library API.
    
    Args:
        query: Search term (book title, author, or subject)
        limit: Maximum number of results to return
    
    Returns:
        List of book dictionaries with metadata
    """
    books = []
    
    try:
        # API endpoint for searching
        url = f"{OPEN_LIBRARY_API}/search.json"
        params = {
            'q': query,
            'limit': limit,
            'fields': 'key,title,author_name,first_publish_year,subject'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse results
        for doc in data.get("docs", [])[:limit]:
            title = doc.get("title", "").strip()
            
            # Skip invalid entries
            if not title or len(title) < 2:
                continue
            
            authors = doc.get("author_name", [])
            author = authors[0] if authors else "Unknown Author"
            
            # Fetch additional details
            book_key = doc.get("key", "")
            description = fetch_book_description(book_key)
            
            book_obj = {
                'title': title,
                'author': author,
                'description': description,
                'publish_year': doc.get("first_publish_year", "Unknown"),
                'subjects': doc.get("subject", [])[:3],
                'price': estimate_book_price(title),
                'book_key': book_key,
                'source': 'Open Library',
                'added_date': datetime.now().isoformat()
            }
            
            books.append(book_obj)
        
        logger.info(f"Found {len(books)} books for query: {query}")
        return books
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout. Please try again.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error searching books: {str(e)}")
        logger.error(f"API error: {str(e)}")
        return []
    except json.JSONDecodeError:
        st.error("❌ Invalid response from server")
        return []

def fetch_book_description(book_key: str) -> str:
    """
    Fetch detailed description from Open Library.
    
    Args:
        book_key: Open Library book key (e.g., '/works/OL12345W')
    
    Returns:
        Book description or placeholder text
    """
    if not book_key:
        return "No description available"
    
    try:
        url = f"{OPEN_LIBRARY_API}{book_key}.json"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        desc = data.get("description", {})
        
        # Handle both string and dict descriptions
        if isinstance(desc, dict):
            desc_text = desc.get("value", "")
        else:
            desc_text = str(desc)
        
        if desc_text:
            # Truncate to reasonable length
            return desc_text[:500] + "..." if len(desc_text) > 500 else desc_text
        
        return "No description available"
    
    except Exception as e:
        logger.debug(f"Could not fetch description: {str(e)}")
        return "No description available"

def estimate_book_price(title: str) -> float:
    """
    Estimate book price based on category.
    Applies student discount if applicable.
    
    Args:
        title: Book title for category detection
    
    Returns:
        Estimated price as float
    """
    title_lower = title.lower()
    
    # Categorize and estimate base price
    if any(word in title_lower for word in ['textbook', 'handbook', 'manual', 'guide', 'reference']):
        base_price = random.uniform(25.99, 79.99)
    elif any(word in title_lower for word in ['novel', 'fiction', 'story', 'mystery', 'romance']):
        base_price = random.uniform(8.99, 16.99)
    elif any(word in title_lower for word in ['biography', 'history', 'science', 'technology', 'physics']):
        base_price = random.uniform(12.99, 24.99)
    else:
        base_price = random.uniform(9.99, 29.99)
    
    # Apply student discount
    if st.session_state.user_profile.get('is_student'):
        base_price *= (1 - STUDENT_DISCOUNT)
    
    return round(base_price, 2)

# ============================================================================
# AI FUNCTIONS
# ============================================================================

def generate_ai_summary(book_title: str, book_author: str) -> str:
    """
    Generate AI-powered book summary using Hugging Face model.
    
    Args:
        book_title: Title of the book
        book_author: Author of the book
    
    Returns:
        AI-generated summary or error message
    """
    if not ai_model:
        return "🔌 AI model not available. Please configure HF_TOKEN in Streamlit secrets."
    
    try:
        prompt = f"Write a brief, engaging summary of '{book_title}' by {book_author}:"
        
        response = ai_model(
            prompt,
            max_length=SUMMARY_LENGTH,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
        
        summary = response[0]['generated_text'].replace(prompt, "").strip()
        
        # Clean and truncate
        summary = summary.split('\n')[0]  # Take first line
        return summary[:SUMMARY_LENGTH] + "..." if len(summary) > SUMMARY_LENGTH else summary
    
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        return "Could not generate AI summary at this moment."

def get_ai_recommendations(book_title: str, is_student: bool = False) -> list:
    """
    Get AI-powered book recommendations based on selected book.
    
    Args:
        book_title: Title of the reference book
        is_student: Whether to filter recommendations for students
    
    Returns:
        List of recommended books
    """
    if not ai_model:
        return get_keyword_based_suggestions(book_title)
    
    try:
        context = "books suitable for students" if is_student else "similar books"
        prompt = f"Recommend 3 {context} similar to '{book_title}':"
        
        response = ai_model(
            prompt,
            max_length=100,
            num_return_sequences=1,
            do_sample=True
        )
        
        suggestions_text = response[0]['generated_text'].replace(prompt, "").strip()
        
        # Parse into list
        lines = [line.strip() for line in suggestions_text.split('\n') if line.strip()]
        return lines[:3] if lines else get_keyword_based_suggestions(book_title)
    
    except Exception as e:
        logger.error(f"Recommendations failed: {str(e)}")
        return get_keyword_based_suggestions(book_title)

def get_keyword_based_suggestions(book_title: str) -> list:
    """
    Fallback: Get recommendations based on keyword matching.
    
    Args:
        book_title: Title of the reference book
    
    Returns:
        List of suggested book titles
    """
    title_lower = book_title.lower()
    
    suggestions = {
        'programming': [
            "Clean Code by Robert Martin",
            "The Pragmatic Programmer",
            "Python Crash Course by Eric Matthes"
        ],
        'fiction': [
            "The Great Gatsby by F. Scott Fitzgerald",
            "To Kill a Mockingbird by Harper Lee",
            "1984 by George Orwell"
        ],
        'science': [
            "A Brief History of Time by Stephen Hawking",
            "The Selfish Gene by Richard Dawkins",
            "Cosmos by Carl Sagan"
        ],
        'biography': [
            "Steve Jobs by Walter Isaacson",
            "Educated by Tara Westover",
            "The Diary of Anne Frank"
        ]
    }
    
    # Find matching category
    for category, books in suggestions.items():
        if category in title_lower:
            return books
    
    return [
        "The Midnight Library by Matt Haig",
        "Atomic Habits by James Clear",
        "Thinking, Fast and Slow by Daniel Kahneman"
    ]

def get_student_book_recommendations(age: int) -> list:
    """
    Get age-appropriate book recommendations for students.
    
    Args:
        age: Age of the student
    
    Returns:
        List of recommended books with descriptions
    """
    if age < 18:  # High school
        return [
            {
                "title": "SAT Prep Guide 2024",
                "reason": "📝 Essential for college admissions",
                "query": "SAT preparation"
            },
            {
                "title": "To Kill a Mockingbird",
                "reason": "📖 Classic literature curriculum",
                "query": "Harper Lee fiction"
            },
            {
                "title": "The Catcher in the Rye",
                "reason": "🎓 High school literature",
                "query": "J.D. Salinger"
            }
        ]
    else:  # College/University
        return [
            {
                "title": "Introduction to Algorithms",
                "reason": "💻 Standard CS textbook",
                "query": "algorithms data structures"
            },
            {
                "title": "Calculus Textbook",
                "reason": "🔢 Essential math resource",
                "query": "calculus mathematics"
            },
            {
                "title": "Academic Writing Guide",
                "reason": "✍️ Improve research papers",
                "query": "academic writing"
            }
        ]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def add_to_search_history(query: str):
    """Add search query to history."""
    if query and query not in st.session_state.search_history:
        st.session_state.search_history.insert(0, query)
        # Keep only last 20 searches
        st.session_state.search_history = st.session_state.search_history[:20]

def add_to_favorites(book: dict):
    """Add book to favorites."""
    # Check if already favorited
    if not any(b['title'] == book['title'] for b in st.session_state.favorites):
        st.session_state.favorites.append(book)
        st.success("✅ Added to favorites!")
        return True
    else:
        st.warning("⚠️ Already in favorites")
        return False

def remove_from_favorites(book_title: str):
    """Remove book from favorites."""
    st.session_state.favorites = [
        b for b in st.session_state.favorites 
        if b['title'] != book_title
    ]

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_user_profile_sidebar():
    """Render user profile configuration in sidebar."""
    st.sidebar.subheader("👤 Your Profile")
    
    age = st.sidebar.number_input(
        "Your Age",
        min_value=5,
        max_value=100,
        value=st.session_state.user_profile['age'],
        step=1
    )
    
    is_student = st.sidebar.checkbox(
        "🎓 I'm a student",
        value=st.session_state.user_profile['is_student']
    )
    
    if st.sidebar.button("💾 Save Profile", type="primary", use_container_width=True):
        st.session_state.user_profile['age'] = age
        st.session_state.user_profile['is_student'] = is_student
        st.session_state.user_profile['profile_saved'] = True
        st.toast("✅ Profile saved successfully!", icon="🎉")
    
    # Display saved profile
    if st.session_state.user_profile['profile_saved']:
        st.sidebar.divider()
        st.sidebar.write("**✅ Current Profile:**")
        st.sidebar.write(f"• **Age:** {st.session_state.user_profile['age']} years")
        st.sidebar.write(
            f"• **Student:** {'🎓 Yes' if st.session_state.user_profile['is_student'] else '❌ No'}"
        )
        
        if st.session_state.user_profile['is_student']:
            st.sidebar.write(f"• **Discount:** 25% Student Discount ✨")

def render_student_recommendations_sidebar():
    """Render student book recommendations in sidebar."""
    if not st.session_state.user_profile.get('is_student'):
        return
    
    st.sidebar.divider()
    st.sidebar.subheader("🎓 Student Picks")
    
    age = st.session_state.user_profile['age']
    recommendations = get_student_book_recommendations(age)
    
    for rec in recommendations:
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.write(f"**{rec['title']}**")
            st.caption(rec['reason'])
        
        with col2:
            if st.button("🔍", key=f"student_rec_{rec['title']}", help="Search this book"):
                st.session_state.current_search_results = search_open_library(rec['query'])
                st.session_state.selected_book = None
                st.rerun()

def render_search_history_sidebar():
    """Render search history in sidebar."""
    if not st.session_state.search_history:
        return
    
    st.sidebar.divider()
    st.sidebar.subheader("📜 Search History")
    
    for i, query in enumerate(st.session_state.search_history[:10]):
        col1, col2 = st.sidebar.columns([3, 1])
        
        with col1:
            if st.button(f"🔍 {query}", key=f"history_{i}", use_container_width=True):
                st.session_state.current_search_results = search_open_library(query)
                st.session_state.selected_book = None
                st.rerun()
        
        with col2:
            if st.button("✕", key=f"del_history_{i}", help="Remove from history"):
                st.session_state.search_history.pop(i)
                st.rerun()

def render_favorites_sidebar():
    """Render favorites list in sidebar."""
    if not st.session_state.favorites:
        return
    
    st.sidebar.divider()
    st.sidebar.subheader(f"❤️ Favorites ({len(st.session_state.favorites)})")
    
    for book in st.session_state.favorites[:5]:
        col1, col2 = st.sidebar.columns([3, 1])
        
        with col1:
            if st.button(
                f"📖 {book['title'][:20]}...",
                key=f"fav_{book['title']}",
                use_container_width=True
            ):
                st.session_state.selected_book = book
                st.rerun()
        
        with col2:
            if st.button("✕", key=f"remove_fav_{book['title']}", help="Remove"):
                remove_from_favorites(book['title'])
                st.rerun()

def render_book_card(book: dict, index: int):
    """Render individual book search result card."""
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"#{index + 1} {book['title'][:40]}")
            st.write(f"**By:** {book['author']}")
            st.caption(f"📅 {book['publish_year']} • 📚 {book['source']}")
        
        with col2:
            st.metric("Price", f"${book['price']}")
            if st.session_state.user_profile.get('is_student'):
                st.caption(f"Student: ${book['price'] * 0.75:.2f} 🎓")
        
        # Description
        if book['description']:
            st.write(f"_{book['description'][:200]}..._")
        
        # Subjects
        if book.get('subjects'):
            subjects = ", ".join(book['subjects'][:2])
            st.caption(f"📌 {subjects}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(
                "📖 View Details",
                key=f"view_{index}_{book['title']}",
                use_container_width=True
            ):
                st.session_state.selected_book = book
                st.rerun()
        
        with col2:
            if st.button(
                "❤️ Favorite",
                key=f"fav_btn_{index}_{book['title']}",
                use_container_width=True
            ):
                add_to_favorites(book)
        
        with col3:
            if st.button(
                "🔗 More Info",
                key=f"info_{index}_{book['title']}",
                use_container_width=True
            ):
                st.info(f"Book Key: {book.get('book_key', 'N/A')}")

def render_book_detail_view(book: dict):
    """Render detailed view of selected book."""
    st.subheader(f"📖 {book['title']}")
    st.divider()
    
    # Main info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Author", book['author'])
    
    with col2:
        st.metric("Year", book['publish_year'])
    
    with col3:
        st.metric("Price", f"${book['price']}")
    
    # Description
    if book['description']:
        st.write("### 📝 Description")
        st.info(book['description'])
    
    # Subjects
    if book.get('subjects'):
        st.write("### 📌 Categories")
        cols = st.columns(len(book['subjects']))
        for col, subject in zip(cols, book['subjects']):
            with col:
                st.write(f"🏷️ {subject}")
    
    st.divider()
    
    # AI Summary section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🤖 Generate AI Summary", use_container_width=True):
            with st.spinner("✨ AI is creating a summary..."):
                summary = generate_ai_summary(book['title'], book['author'])
                st.session_state.ai_summary = summary
                st.rerun()
    
    if st.session_state.ai_summary:
        st.write("### 🤖 AI-Generated Summary")
        st.success(st.session_state.ai_summary)
    
    st.divider()
    
    # Recommendations
    st.write("### 💡 You Might Also Like")
    
    with st.spinner("🔄 Finding related books..."):
        recommendations = get_ai_recommendations(
            book['title'],
            st.session_state.user_profile.get('is_student', False)
        )
    
    cols = st.columns(len(recommendations))
    for col, rec in zip(cols, recommendations):
        with col:
            st.write(f"**{rec}**")
            if st.button(
                "🔍 Search",
                key=f"search_rec_{rec}",
                use_container_width=True
            ):
                st.session_state.current_search_results = search_open_library(rec)
                st.session_state.selected_book = None
                st.rerun()
    
    st.divider()
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("❤️ Add to Favorites", use_container_width=True):
            add_to_favorites(book)
    
    with col2:
        if st.button("🔍 New Search", use_container_width=True):
            st.session_state.selected_book = None
            st.session_state.ai_summary = ""
            st.rerun()
    
    with col3:
        if st.button("📋 Back to Results", use_container_width=True):
            st.session_state.selected_book = None
            st.rerun()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        render_user_profile_sidebar()
        render_student_recommendations_sidebar()
        render_search_history_sidebar()
        render_favorites_sidebar()
        
        # About section
        st.divider()
        with st.expander("ℹ️ About"):
            st.write(
                """
                **AI Library Assistant v2.0**
                
                🎯 Find and discover books with AI-powered insights.
                
                ✨ Features:
                - 📚 Search across millions of books
                - 🤖 AI-generated summaries
                - 💡 Smart recommendations
                - 🎓 Student discounts
                - ❤️ Save favorites
                
                📖 Data from Open Library API
                """
            )
    
    # Main content
    st.title("🤖 AI Library Assistant")
    st.markdown("### Discover Books. Get AI Insights. Find Your Next Read.")
    st.divider()
    
    # Check if user profile is saved
    if not st.session_state.user_profile['profile_saved']:
        st.warning("⚠️ Please save your profile in the sidebar to get personalized recommendations!")
    
    # Selected book detail view
    if st.session_state.selected_book:
        render_book_detail_view(st.session_state.selected_book)
    
    # Search section
    else:
        st.subheader("🔍 Search Books")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search by book title, author, or topic:",
                placeholder="e.g., 'Python programming', 'Science fiction', 'Stephen King'",
                help="Enter keywords to search Open Library"
            )
        
        with col2:
            search_clicked = st.button(
                "🚀 Search",
                type="primary",
                use_container_width=True,
                help="Click to search"
            )
        
        # Perform search
        if search_clicked and search_query:
            add_to_search_history(search_query)
            
            with st.spinner("🔍 Searching Open Library..."):
                st.session_state.current_search_results = search_open_library(search_query)
            
            st.rerun()
        
        # Display search results
        if st.session_state.current_search_results:
            st.divider()
            st.subheader(
                f"📚 Found {len(st.session_state.current_search_results)} Books"
            )
            
            # Results in grid
            for i, book in enumerate(st.session_state.current_search_results):
                render_book_card(book, i)
        
        # Empty state
        elif not search_query:
            st.divider()
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(
                    """
                    ### 👋 Welcome!
                    
                    1. **Set your profile** in the sidebar
                    2. **Search** for any book using the search box
                    3. **Explore** detailed information and AI-generated summaries
                    4. **Save** your favorite books
                    
                    **💡 Try searching for:**
                    """
                )
            
            # Example searches
            with col2:
                examples = [
                    ("🐍 Python", "Python programming"),
                    ("🚀 Science", "Science fiction"),
                    ("📚 Classic", "Classic literature"),
                    ("🎓 Learn", "Educational books")
                ]
                
                for emoji_title, query in examples:
                    if st.button(emoji_title, use_container_width=True):
                        st.session_state.current_search_results = search_open_library(query)
                        add_to_search_history(query)
                        st.rerun()
        
        else:
            st.info("No books found. Try a different search term.")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ai_model:
            st.caption("✅ AI Enabled - Summaries Available")
        else:
            st.caption("⚠️ AI Not Available - Configure HF_TOKEN")
    
    with col2:
        st.caption(f"📖 {len(st.session_state.favorites)} Favorites Saved")
    
    with col3:
        st.caption(f"🔍 {len(st.session_state.search_history)} Searches in History")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

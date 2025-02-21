import os
import re
import json
import requests
import spacy
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from urllib.parse import urlparse
from spacy.matcher import PhraseMatcher
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import cpu_count
from io import BytesIO
from pdfminer.high_level import extract_text
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import browser_cookie3
from loguru import logger
from sys import stdout

# Set up Loguru logging with color.
logger.remove()

logger.add(lambda msg: print(msg, end=""), level="INFO", colorize=True,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>")

# Define unwanted tags to filter out.
unwanted_tags = {
    "connect",
    "connect world",
    "world communities",
    "account",
    "sign",
    "sign in",
    "seriously available",
    "seriously",
    "continue forms",
    "forms forgot",
    "suspicious",
    "does",
    "suspicious report",
    "look suspicious",
    "form look",
    "look",
    "form",
    "report",
    "does form",
    "want",
}

# Check for GPU and enable it for spaCy (Apple silicon: uses MPS if available)
try:
    spacy.require_gpu()
    logger.info("GPU enabled for spaCy!")
except Exception as e:
    logger.info("GPU not available. Running on CPU. {}", e)

# Load transformer-based model (more accurate, heavier)
nlp = spacy.load("en_core_web_trf")
matcher = PhraseMatcher(nlp.vocab)

# Disable insecure request warnings.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define topic-based phrase matching patterns.
topics = {
    "cool-ass-math": ["math", "arxiv", "proof", "theorem", "set-theory"],
    "math-blogs": ["math-blog", "mathematical writing", "math discussion"],
    "latex": ["latex", "tex", "typography"],
    "cs-general": ["computer science", "computing", "cs fundamentals"],
    "exciting-cs-developments": ["new technology", "latest cs research", "tech breakthrough"],
    "programming-languages": ["language design", "compiler", "syntax", "parsing"],
    "low-level": ["assembly", "bitwise", "performance optimization"],
    "security-cryptography": ["security", "encryption", "cybersecurity", "hashing"],
    "computer-graphics": ["graphics", "rendering", "ray tracing", "shaders"],
    "machine-learning": ["ml", "deep learning", "neural network", "ai"],
    "cs-tech-concerns": ["tech ethics", "big tech", "privacy"],
    "cs-tooling-or-sites": ["github", "vim", "neovim", "emacs", "editor"],
    "cs-other": ["miscellaneous cs", "interesting cs topics"],
    "cs-courses": ["cs education", "computer science courses", "learning cs"],
    "tutorials": ["tutorial", "how-to", "beginner guide"],
    "rust": ["rustlang", "rust programming"],
    "cs-blogs": ["cs-blog", "tech blog", "programming insights"],
    "books": ["book", "reading", "library"],
    "memes": ["meme", "funny", "tenor", "gif"],
    "talks-videos": ["youtube", "twitch", "talk", "lecture", "conference"],
    "career-and-student-resources": ["internship", "career", "resume", "student"],
    "linguistics": ["phonetics", "syntax", "morphology", "linguistics"],
    "machine-learning-ai": ["ml", "deep learning", "neural network", "ai"],
    "rice-stuff": ["Rice", "campus", "general", "Rice University"],
    "urandom": ["random", "generator", "unix", "entropy"],
    "cryptocurrency": ["crypto", "blockchain", "bitcoin", "ethereum", "altcoin"],
    "cs-questions": ["computer science", "questions", "problems", "discussion"],
    "cs-theory": ["theory", "algorithms", "computation theory", "formal methods", "cs", "computing"],
    "hooman-languages": ["language", "human", "linguistics", "communication"],
    "im-trying-to-learn": ["learning", "self-improvement", "education", "tutorial"],
    "linux": ["linux", "unix", "operating system", "opensource"],
    "student-resources": ["student", "resources", "education", "college", "learning"],
    "general": ["general", "miscellaneous", "discussion", "variety"],
    "life-things": ["life", "personal", "stories", "experiences"],
    "books-and-linguistics": ["book", "reading", "library", "phonetics", "syntax", "morphology", "linguistics"],
    "competitive-programming": ["competitive programming", "coding competitions", "algorithms", "challenges"],
    "text-editors": ["text editor", "vim", "emacs", "sublime", "editor"],
}

# The allowed (predefined) topics are the keys from the topics dictionary.
allowed_topics = set(topics.keys())

# Add phrase matching patterns.
for category, phrases in topics.items():
    patterns = [nlp(text) for text in phrases]
    matcher.add(category, patterns)

def validate_url(url):
    """Clean the URL and ensure it has a proper scheme."""
    url = url.strip().strip('"').strip("'").strip("<>")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    return url

def clean_text(text):
    """Normalize whitespace and strip extra spaces."""
    if text:
        return " ".join(text.split())
    return ""

def sanitize_filename(name):
    """Replaces any invalid filename characters with an underscore."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def setup_http_session():
    """Creates a session with a retry strategy and loads cookies from Firefox."""
    session = requests.Session()
    try:
        session.cookies = browser_cookie3.firefox()
        logger.info("Loaded cookies from Firefox!")
    except Exception as e:
        logger.warning("Could not load Firefox cookies: {}", e)
    retries = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # Use headers similar to Firefox to avoid 406 errors.
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:112.0) Gecko/20100101 Firefox/112.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    })
    return session

def extract_links(filename):
    """Extracts and validates links from the given file."""
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            for url in re.findall(r"https?://\S+", line):
                valid = validate_url(url)
                if valid:
                    yield valid

def sort_links_by_domain(links):
    """Sorts links by domain for optimized session reuse."""
    return sorted(links, key=lambda url: urlparse(url).netloc)

def fetch_page_content(url, session, retries=3):
    """Fetches content from a webpage or PDF."""
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=5, stream=True, verify=False)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "pdf" in content_type or url.endswith(".pdf"):
                return extract_pdf_text(response)
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string.strip() if soup.title else "No Title"
            paragraphs = soup.find_all("p")
            text_content = " ".join(clean_text(p.get_text()) for p in paragraphs[:15])
            return clean_text(title), text_content if text_content else None
        except requests.RequestException as e:
            logger.warning("Request error for {}: {}", url, e)
            if attempt < retries - 1:
                continue
            return None, None

def extract_pdf_text(response):
    """Extracts text from a PDF response.

    First, it attempts to use pdfminer; if that fails, it falls back on PyMuPDF (fitz).
    """
    try:
        with BytesIO(response.content) as pdf_file:
            pdf_text = extract_text(pdf_file).strip()
        if pdf_text:
            return "PDF Document", clean_text(pdf_text)
        else:
            raise ValueError("No text extracted using pdfminer.")
    except Exception as e:
        logger.warning("PDF extraction error using pdfminer: {}", e)
        try:
            import fitz  # PyMuPDF
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            if text:
                return "PDF Document", clean_text(text)
            else:
                raise ValueError("No text extracted using PyMuPDF.")
        except Exception as e2:
            logger.error("Fallback PDF extraction error: {}", e2)
            return "Unreadable PDF", None

def compute_global_tfidf(texts):
    """Builds a global TF-IDF model on all document texts."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def extract_top_keywords_from_vector(row, vectorizer, n=10, min_length=3):
    """Extracts top N keywords from a TF-IDF row filtering by minimum length."""
    row_array = row.toarray().flatten()
    top_indices = row_array.argsort()[::-1][:n]
    feature_names = vectorizer.get_feature_names_out()
    top_keywords = [
        feature_names[i]
        for i in top_indices
        if row_array[i] > 0 and len(feature_names[i]) >= min_length
    ]
    return top_keywords

def domain_based_categories(url):
    """Assigns categories based on the URL's domain."""
    domain = urlparse(url).netloc
    categories = set()
    if "github.com" in domain:
        categories.add("cs-tooling-or-sites")
    elif "arxiv.org" in domain:
        categories.add("cool-ass-math")
    elif "youtube.com" in domain:
        categories.add("talks-videos")
    return categories

def process_all_links(links):
    """Fetches page content concurrently using ThreadPoolExecutor."""
    results = []
    session = setup_http_session()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_page_content, link, session): link for link in links}
        for future in futures:
            link = futures[future]
            title, content = future.result()
            if title and content:
                results.append((link, title, content))
    return results

def batch_nlp_processing(docs):
    """
    Processes texts in batch using spaCy's pipe.
    When using GPU with en_core_web_trf, n_process is forced to 1.
    """
    n_process = 1 if spacy.prefer_gpu() else cpu_count()
    return list(nlp.pipe(docs, n_process=n_process))

def filter_unwanted_tags(tags, unwanted):
    """Return a new set of tags, filtering out any that contain unwanted phrases."""
    filtered = set()
    for tag in tags:
        tag_clean = tag.lower().strip()
        if any(unwanted_phrase in tag_clean for unwanted_phrase in unwanted):
            continue
        filtered.add(tag)
    return filtered

def categorize_documents(results, processed_docs, tfidf_matrix, vectorizer):
    """
    For each link, compute assigned tags using NER, phrase matching,
    TF-IDF, and domain heuristics. Then, select the top 25 dynamic categories
    that are not in allowed topics and not unwanted. Finally, for each link,
    keep only tags that are either predefined or among these top new dynamic tags.
    """
    dynamic_categories = Counter()
    link_categories = []

    for i, (link, title, content) in enumerate(results):
        doc = processed_docs[i]
        assigned = set()

        # NER-based: include allowed labels.
        for ent in doc.ents:
            if ent.label_ in ["EVENT", "WORK_OF_ART", "LAW", "NORP", "LANGUAGE", "ORG", "GPE", "PERSON"]:
                assigned.add(ent.text.lower())

        # Phrase matching.
        matches = matcher(doc)
        for match_id, start, end in matches:
            assigned.add(nlp.vocab.strings[match_id])

        # Global TF-IDF keywords.
        top_keywords = extract_top_keywords_from_vector(tfidf_matrix[i], vectorizer, n=10)
        assigned.update(top_keywords)
        for kw in top_keywords:
            dynamic_categories[kw] += 1

        # Domain-based heuristics.
        assigned.update(domain_based_categories(results[i][0]))

        link_categories.append((link, title, assigned))

    # Compute top 25 dynamic categories not in allowed topics and not unwanted.
    top_new = {tag for tag, count in dynamic_categories.most_common(25)
               if tag not in allowed_topics and tag not in unwanted_tags}

    # For each link, filter its tags.
    categorized_links = defaultdict(list)
    for link, title, assigned in link_categories:
        filtered_tags = filter_unwanted_tags(assigned, unwanted_tags)
        final_tags = {tag for tag in filtered_tags if tag in allowed_topics or tag in top_new}
        if not final_tags:
            final_tags.add("miscellaneous")
        for tag in final_tags:
            categorized_links[tag].append({
                "url": link,
                "title": title,
                "tags": list(final_tags)
            })

    return categorized_links, dynamic_categories

def save_categorized_links(categorized_links):
    """Saves categorized links to JSON files in the 'categorized_links' folder."""
    os.makedirs("categorized_links", exist_ok=True)
    for category, links in categorized_links.items():
        safe_category = sanitize_filename(category)
        filepath = os.path.join("categorized_links", f"{safe_category}.json")
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(links, file, indent=4)

def test_pdf_output(source):
    """
    Tests PDF extraction. If 'source' is a URL (starting with "http"),
    it downloads the PDF; otherwise, it treats 'source' as a local file path.
    """
    logger.info("Testing PDF extraction for: {}", source)
    try:
        if source.lower().startswith("http"):
            response = requests.get(source, timeout=10, verify=False)
            response.raise_for_status()
            content = response.content
        else:
            with open(source, "rb") as f:
                content = f.read()
        # Create a dummy response-like object with a 'content' attribute.
        DummyResponse = type("DummyResponse", (object,), {"content": content})
        title, text = extract_pdf_text(DummyResponse())
        logger.info("Extracted title: {}", title)
        if text:
            logger.info("Extracted text (first 500 chars):\n{}", text[:500])
        else:
            logger.error("No text extracted from PDF.")
    except Exception as e:
        logger.error("Error testing PDF extraction: {}", e)

def main():
    filename = "links.txt"  # Update this to point to your links file.
    links = list(extract_links(filename))
    sorted_links = sort_links_by_domain(links)
    logger.info("Total links to process: {}", len(sorted_links))

    # Fetch page content concurrently.
    fetched_results = process_all_links(sorted_links)
    if not fetched_results:
        logger.error("No links fetched successfully.")
        return

    # Combine title and content for NLP processing.
    docs = [f"{title} {content}" for _, title, content in fetched_results]
    docs = [clean_text(doc) for doc in docs]

    # Batch process documents using spaCy.
    processed_docs = batch_nlp_processing(docs)

    # Use full content for global TF-IDF.
    texts = [clean_text(content) for _, _, content in fetched_results]
    tfidf_matrix, vectorizer = compute_global_tfidf(texts)

    # Categorize documents.
    categorized_links, dynamic_categories = categorize_documents(
        fetched_results, processed_docs, tfidf_matrix, vectorizer
    )

    # Save categorized links.
    save_categorized_links(categorized_links)
    # Filter dynamic categories for logging.
    filtered_dynamic = Counter({tag: count for tag, count in dynamic_categories.items() if tag not in unwanted_tags})
    filtered_top_new = [(tag, count) for tag, count in dynamic_categories.most_common(25)
                        if tag not in allowed_topics and tag not in unwanted_tags]

    logger.info("Categorization complete. Check the 'categorized_links' folder.")
    logger.info("Dynamic categories (filtered): {}", filtered_dynamic.most_common(10))
    logger.info("Top new dynamic categories (filtered): {}", filtered_top_new)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "testpdf":
        source = sys.argv[2] if len(sys.argv) > 2 else "sample.pdf"
        test_pdf_output(source)
    else:
        main()

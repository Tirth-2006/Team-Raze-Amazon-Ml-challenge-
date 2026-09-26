"""
Text normalization for business names and addresses
Conservative approach - preserve original signals while adding normalized variants
"""
import re
from unidecode import unidecode


# Common business legal suffixes and abbreviations
LEGAL_SUFFIX_MAP = {
    # English
    'corporation': 'corp', 'incorporated': 'inc', 'company': 'co',
    'limited': 'ltd', 'private': 'pvt', 'public': 'pub',
    'limited liability company': 'llc', 'limited liability partnership': 'llp',
    'professional corporation': 'pc', 'professional limited liability company': 'pllc',
    # Common variations
    'corp.': 'corp', 'inc.': 'inc', 'co.': 'co', 'ltd.': 'ltd',
    'pvt.': 'pvt', 'llc.': 'llc', 'llp.': 'llp',
    # Indian specific
    'private limited': 'pvt ltd', 'प्राइवेट लिमिटेड': 'pvt ltd',
    'limited': 'ltd', 'लिमिटेड': 'ltd',
}


ADDRESS_ABBREV_MAP = {
    # Street types
    'street': 'st', 'road': 'rd', 'avenue': 'ave', 'drive': 'dr',
    'boulevard': 'blvd', 'lane': 'ln', 'court': 'ct', 'circle': 'cir',
    'place': 'pl', 'square': 'sq', 'terrace': 'ter', 'highway': 'hwy',
    'parkway': 'pkwy', 'way': 'way',
    # Directionals
    'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw',
    # Unit types
    'apartment': 'apt', 'suite': 'ste', 'unit': 'unit', 'floor': 'fl',
    'building': 'bldg', 'room': 'rm',
}


def normalize_business_name(name: str) -> str:
    """
    Normalize business name - conservative approach

    Steps:
    1. Transliterate non-ASCII to ASCII
    2. Lowercase
    3. Remove punctuation except spaces
    4. Normalize legal suffixes
    5. Remove extra whitespace
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    # Transliterate (handles Hindi/Kannada/French characters)
    text = unidecode(name)

    # Lowercase
    text = text.lower()

    # Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalize legal suffixes
    for full, abbrev in LEGAL_SUFFIX_MAP.items():
        # Word boundary match to avoid partial matches
        text = re.sub(r'\b' + re.escape(full) + r'\b', abbrev, text)

    # Collapse whitespace
    text = ' '.join(text.split())

    return text.strip()


def normalize_address(address: str) -> str:
    """
    Normalize address - very conservative given open country set

    Steps:
    1. Transliterate non-ASCII
    2. Lowercase
    3. Remove punctuation except spaces and commas
    4. Normalize common street abbreviations
    5. Remove extra whitespace
    """
    if not isinstance(address, str) or not address.strip():
        return ""

    # Transliterate
    text = unidecode(address)

    # Lowercase
    text = text.lower()

    # Remove most punctuation but keep commas (structural) and spaces
    text = re.sub(r'[^\w\s,]', ' ', text)

    # Normalize street abbreviations - only safe common ones
    for full, abbrev in ADDRESS_ABBREV_MAP.items():
        text = re.sub(r'\b' + re.escape(full) + r'\b', abbrev, text)

    # Collapse whitespace
    text = ' '.join(text.split())

    return text.strip()


def extract_tokens(text: str) -> set:
    """
    Extract token set from normalized text (for Jaccard similarity)
    """
    if not text:
        return set()
    return set(text.split())


def extract_numeric_tokens(text: str) -> set:
    """
    Extract numeric tokens from text (addresses often have key numbers)
    """
    if not text:
        return set()
    return set(re.findall(r'\d+', text))

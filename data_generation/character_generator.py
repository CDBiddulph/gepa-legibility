"""
Character generator that creates unique character profiles from data files.

Uses pre-existing lists of first names, last names, and occupations to generate
character profiles without any LLM calls.
"""

import os
import random
from typing import List, Dict, Any, Tuple

import pandas as pd


# Data file paths (relative to this file's directory)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIRST_NAMES_FILE = os.path.join(DATA_DIR, "first_names.txt")
LAST_NAMES_FILE = os.path.join(DATA_DIR, "last_names.xls")
OCCUPATIONS_FILE = os.path.join(DATA_DIR, "occupations.xlsx")

# Age range for generated characters
MIN_AGE = 18
MAX_AGE = 75


def load_first_names() -> List[str]:
    """Load first names from CSV file.

    File format: name,gender,count
    """
    names = []
    with open(FIRST_NAMES_FILE, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if parts:
                name = parts[0].strip()
                if name:
                    # Title case the name
                    names.append(name.title())
    return names


def load_last_names() -> List[str]:
    """Load last names from Excel file."""
    df = pd.read_excel(LAST_NAMES_FILE, skiprows=1)
    # Names are in the 'name' column, in uppercase
    names = df["name"].dropna().tolist()
    # Convert to title case
    return [name.title() for name in names if isinstance(name, str)]


def load_occupations() -> List[str]:
    """Load occupations from Excel file."""
    df = pd.read_excel(OCCUPATIONS_FILE, engine="openpyxl", skiprows=5)
    # Occupations are in the first column
    col_name = df.columns[0]
    occupations = df[col_name].dropna().tolist()
    # Filter out non-string entries and clean up
    cleaned = []
    for occ in occupations:
        if isinstance(occ, str):
            occ = occ.strip()
            # Skip entries that look like codes or are too short
            if len(occ) > 2 and not occ.replace("-", "").replace(" ", "").isdigit():
                cleaned.append(occ)
    return cleaned


class CharacterGenerator:
    """Generates unique character profiles from pre-loaded data files."""

    def __init__(self, seed: int = None):
        """Initialize the generator by loading all data files.

        Args:
            seed: Random seed for reproducibility. If provided, generation is
                  fully deterministic across runs.
        """
        print("Loading character data files...")
        self.first_names = load_first_names()
        self.last_names = load_last_names()
        self.occupations = load_occupations()

        print(f"  Loaded {len(self.first_names)} first names")
        print(f"  Loaded {len(self.last_names)} last names")
        print(f"  Loaded {len(self.occupations)} occupations")

        # Track used combinations to ensure uniqueness
        self.used_names: set = set()

        # Use a dedicated Random instance for isolation and determinism
        # This ensures other code using the global random module won't affect us
        self._seed = seed
        self._rng = random.Random(seed)

    def _generate_unique_name(self) -> str:
        """Generate a unique full name that hasn't been used yet."""
        max_attempts = 1000
        for _ in range(max_attempts):
            first = self._rng.choice(self.first_names)
            last = self._rng.choice(self.last_names)
            full_name = f"{first} {last}"

            # Check for uniqueness (case-insensitive)
            if full_name.lower() not in self.used_names:
                self.used_names.add(full_name.lower())
                return full_name

        raise RuntimeError(
            f"Failed to generate unique name after {max_attempts} attempts. "
            f"Used {len(self.used_names)} names so far."
        )

    def _generate_age(self) -> int:
        """Generate a random age."""
        return self._rng.randint(MIN_AGE, MAX_AGE)

    def _generate_occupation(self) -> str:
        """Generate a random occupation."""
        return self._rng.choice(self.occupations)

    def generate_character(self) -> Dict[str, Any]:
        """Generate a single character profile.

        Returns:
            Dict with keys: name, age, job
        """
        return {
            "name": self._generate_unique_name(),
            "age": self._generate_age(),
            "job": self._generate_occupation(),
        }

    def generate_characters(self, count: int) -> List[Dict[str, Any]]:
        """Generate multiple unique character profiles.

        Args:
            count: Number of characters to generate

        Returns:
            List of character dicts, each with keys: name, age, job
        """
        characters = []
        for i in range(count):
            characters.append(self.generate_character())
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{count} characters")
        return characters

    def reset(self):
        """Reset the generator to its initial state for reproducible regeneration."""
        self.used_names.clear()
        self._rng = random.Random(self._seed)


def generate_unique_characters(count: int, seed: int = None) -> List[Dict[str, Any]]:
    """Convenience function to generate unique character profiles.

    Args:
        count: Number of characters to generate
        seed: Random seed for reproducibility

    Returns:
        List of character dicts, each with keys: name, age, job
    """
    generator = CharacterGenerator(seed=seed)
    return generator.generate_characters(count)

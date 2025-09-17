import re


def score(expected_solution, attempted_solution):
    """
    Score a puzzle answer against the solution.

    Args:
        expected_solution: String containing the expected solution (like from JSONL "solution" field)
        attempted_solution: String containing the attempted solution

    Returns:
        float: Score from 0.0 to 1.0 representing percentage of correct squares
    """
    # Parse expected solution into sections
    expected_grid, expected_across, expected_down = _parse_expected_solution(
        expected_solution
    )

    # If attempted solution is empty, score is 0.0
    content = attempted_solution.strip()
    if not content:
        return 0.0, 0, 0

    # Split by double newlines to get pieces
    pieces = content.split("\n\n")
    if not pieces or all(not piece.strip() for piece in pieces):
        return 0.0, 0, 0

    # Track correctness of each fillable square by position in grid
    # We'll use row,col coordinates since that's easier to work with
    width = len(expected_grid[0]) if expected_grid else 0
    height = len(expected_grid)

    if width == 0 or height == 0:
        return 0.0, 0, 0

    # Track which squares are correct/incorrect
    correct_squares = set()
    incorrect_squares = set()

    # Process each piece
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        if piece.startswith("Across:"):
            _process_across_section(
                piece,
                expected_across,
                width,
                expected_grid,
                correct_squares,
                incorrect_squares,
            )
        elif piece.startswith("Down:"):
            _process_down_section(
                piece,
                expected_down,
                height,
                expected_grid,
                correct_squares,
                incorrect_squares,
            )
        else:
            # Assume it's a grid
            _process_grid_section(
                piece, expected_grid, correct_squares, incorrect_squares
            )

    # Count fillable squares and determine final correct squares
    fillable_count = 0
    final_correct_squares = set()

    for row in range(height):
        for col in range(width):
            if expected_grid[row][col] != ".":  # Not a black square
                fillable_count += 1
                pos = (row, col)
                if pos in correct_squares and pos not in incorrect_squares:
                    final_correct_squares.add(pos)

    correct_count = len(final_correct_squares)

    # Calculate score
    if fillable_count == 0:
        return 0.0, 0, 0

    score = correct_count / fillable_count
    return score, correct_count, fillable_count


def _parse_expected_solution(expected_solution):
    """Parse expected solution into grid, across clues, and down clues"""
    lines = expected_solution.strip().split("\n")

    # Extract grid
    grid_lines = []
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if not line or line.startswith("Across:") or line.startswith("Down:"):
            break
        grid_lines.append(line.split())
        line_idx += 1

    # Extract clues
    across_clues = {}
    down_clues = {}

    current_section = None
    for i in range(line_idx, len(lines)):
        line = lines[i].strip()
        if line.startswith("Across:"):
            current_section = "across"
        elif line.startswith("Down:"):
            current_section = "down"
        elif line and current_section:
            # Parse clue line
            match = re.match(r"\s*(\d+)\.\s*(.+)", line)
            if match:
                clue_num = int(match.group(1))
                answer = match.group(2).strip().upper()
                if current_section == "across":
                    across_clues[clue_num] = answer
                else:
                    down_clues[clue_num] = answer

    return grid_lines, across_clues, down_clues


def _process_grid_section(grid_text, expected_grid, correct_squares, incorrect_squares):
    """Process a grid format piece"""
    lines = grid_text.strip().split("\n")
    height = len(expected_grid)
    width = len(expected_grid[0]) if height > 0 else 0

    for row_idx, line in enumerate(lines):
        if row_idx >= height:
            break

        cells = line.split()
        for col_idx, cell in enumerate(cells):
            if col_idx >= width:
                break

            expected = expected_grid[row_idx][col_idx].upper()
            actual = cell.upper()
            pos = (row_idx, col_idx)

            # Skip black squares
            if expected == ".":
                continue

            # Mark square as correct or incorrect
            if actual == expected:
                correct_squares.add(pos)
            else:
                incorrect_squares.add(pos)


def _process_across_section(
    section_text,
    expected_across,
    width,
    expected_grid,
    correct_squares,
    incorrect_squares,
):
    """Process an Across: section"""
    lines = section_text.split("\n")[1:]  # Skip the "Across:" line

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse clue number and answer
        match = re.match(r"\s*(\d+)\.\s*(.+)", line)
        if not match:
            continue

        clue_num = int(match.group(1))
        answer = match.group(2).strip().upper()

        # Find where this clue should be in the expected solution
        if clue_num not in expected_across:
            continue

        expected_answer = expected_across[clue_num]

        # Find the position of this clue in the grid
        clue_pos = _find_clue_position(clue_num, expected_grid, "across")
        if clue_pos is None:
            continue

        row, col = clue_pos

        # Check each letter of the answer
        for i, letter in enumerate(answer):
            if col + i >= width or i >= len(expected_answer):
                break

            pos = (row, col + i)
            expected_letter = expected_answer[i].upper()

            # Skip black squares
            if expected_grid[row][col + i] == ".":
                continue

            # Mark square as correct or incorrect
            if letter == expected_letter:
                correct_squares.add(pos)
            else:
                incorrect_squares.add(pos)


def _process_down_section(
    section_text,
    expected_down,
    height,
    expected_grid,
    correct_squares,
    incorrect_squares,
):
    """Process a Down: section"""
    lines = section_text.split("\n")[1:]  # Skip the "Down:" line

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse clue number and answer
        match = re.match(r"\s*(\d+)\.\s*(.+)", line)
        if not match:
            continue

        clue_num = int(match.group(1))
        answer = match.group(2).strip().upper()

        # Find where this clue should be in the expected solution
        if clue_num not in expected_down:
            continue

        expected_answer = expected_down[clue_num]

        # Find the position of this clue in the grid
        clue_pos = _find_clue_position(clue_num, expected_grid, "down")
        if clue_pos is None:
            continue

        row, col = clue_pos

        # Check each letter of the answer
        for i, letter in enumerate(answer):
            if row + i >= height or i >= len(expected_answer):
                break

            pos = (row + i, col)
            expected_letter = expected_answer[i].upper()

            # Skip black squares
            if expected_grid[row + i][col] == ".":
                continue

            # Mark square as correct or incorrect
            if letter == expected_letter:
                correct_squares.add(pos)
            else:
                incorrect_squares.add(pos)


def _find_clue_position(clue_num, grid, direction):
    """Find the starting position of a clue in the grid based on numbering logic"""
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    current_num = 1

    for row in range(height):
        for col in range(width):
            # Skip black squares
            if grid[row][col] == ".":
                continue

            # Check if this position starts a word
            starts_across = (
                (col == 0 or grid[row][col - 1] == ".")
                and col + 1 < width
                and grid[row][col + 1] != "."
            )
            starts_down = (
                (row == 0 or grid[row - 1][col] == ".")
                and row + 1 < height
                and grid[row + 1][col] != "."
            )

            if starts_across or starts_down:
                if current_num == clue_num:
                    return (row, col)
                current_num += 1

    return None

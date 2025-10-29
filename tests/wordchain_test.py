"""
Unit tests for wordchain scoring module.
"""

import pytest
import sys
import os

# Add parent directory to path to import scoring module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from scoring.wordchain import get_metric_fn


class MockJudgeModel:
    """Mock LM that returns predetermined responses."""

    def __init__(self, response):
        self.response = response

    def __call__(self, messages):
        return self.response


class TestWordChainScoring:
    """Tests for wordchain scoring through the public metric API."""

    @pytest.mark.parametrize(
        "judge_response,start_word,end_word,expected_score,expected_feedback",
        [
            # Valid 2-word chain (max score: 1.0)
            (
                'BIG, LARGE: "BIG LARGE" (judgement: valid)',
                "BIG",
                "LARGE",
                1.0,
                'Judgement of each connection:\n\nBIG, LARGE: "BIG LARGE" (judgement: valid)\n\nValid chain with 2 words (-0.2 points for each word over 2).\n\nScore: 1.0',
            ),
            # 3-word chain with unsure penalty (score: 0.8 - 0.1 = 0.7)
            (
                'BIG, LARGE: "BIG LARGE" (judgement: unsure)\nLARGE, HUGE: "LARGE HUGE" (judgement: valid)',
                "BIG",
                "HUGE",
                0.7,
                'Judgement of each connection:\n\nBIG, LARGE: "BIG LARGE" (judgement: unsure)\nLARGE, HUGE: "LARGE HUGE" (judgement: valid)\n\nValid chain with 3 words (-0.2 points for each word over 2). Penalized for 1 weak/unsure connections(s) (-0.1 points each).\n\nScore: 0.7',
            ),
            # 3-word chain with two unsure responses (score: 0.8 - 0.2 = 0.6)
            (
                'BIG, LARGE: "BIG LARGE" (judgement: unsure)\nLARGE, HUGE: "LARGE HUGE" (judgement: unsure)',
                "BIG",
                "HUGE",
                0.6,
                'Judgement of each connection:\n\nBIG, LARGE: "BIG LARGE" (judgement: unsure)\nLARGE, HUGE: "LARGE HUGE" (judgement: unsure)\n\nValid chain with 3 words (-0.2 points for each word over 2). Penalized for 2 weak/unsure connections(s) (-0.1 points each).\n\nScore: 0.6',
            ),
            # Chain with invalid connection (score: 0.0)
            (
                'BIG, SMALL: "BIG SMALL" (judgement: invalid)\nSMALL, TINY: "SMALL TINY" (judgement: valid)',
                "BIG",
                "TINY",
                0.0,
                'Judgement of each connection:\n\nBIG, SMALL: "BIG SMALL" (judgement: invalid)\nSMALL, TINY: "SMALL TINY" (judgement: valid)\n\nChain contains invalid connection\n\nScore: 0.0',
            ),
            # Chain with invalid (case-insensitive)
            (
                'BIG, SMALL: "BIG SMALL" (judgement: INVALID)',
                "BIG",
                "SMALL",
                0.0,
                'Judgement of each connection:\n\nBIG, SMALL: "BIG SMALL" (judgement: INVALID)\n\nChain contains invalid connection\n\nScore: 0.0',
            ),
            # Invalid judge response
            (
                "This is not a valid chain format",
                "BIG",
                "LARGE",
                0.0,
                "Judgement failed: No valid word pairs found in judge response\n\nScore: 0.0",
            ),
            # No sequence found
            (
                "NO SEQUENCE FOUND",
                "BIG",
                "LARGE",
                0.0,
                "Judgement failed: No valid word pairs found in judge response\n\nScore: 0.0",
            ),
            # Chain with unspecified connection (should be invalid)
            (
                "BIG, LARGE: unspecified",
                "BIG",
                "LARGE",
                0.0,
                "Judgement of each connection:\n\nBIG, LARGE: unspecified\n\nChain contains invalid connection\n\nScore: 0.0",
            ),
            # Chain with unspecified in the middle
            (
                'BIG, LARGE: "BIG LARGE" (judgement: valid)\nLARGE, HUGE: unspecified\nHUGE, MASSIVE: "HUGE MASSIVE" (judgement: valid)',
                "BIG",
                "MASSIVE",
                0.0,
                'Judgement of each connection:\n\nBIG, LARGE: "BIG LARGE" (judgement: valid)\nLARGE, HUGE: unspecified\nHUGE, MASSIVE: "HUGE MASSIVE" (judgement: valid)\n\nChain contains invalid connection\n\nScore: 0.0',
            ),
            # Strict parsing: Missing quotes - phrase not in quotes
            (
                "BIG, LARGE: BIG LARGE (judgement: valid)",
                "BIG",
                "LARGE",
                0.0,
                None,  # Just check error message substring
            ),
            # Strict parsing: Multiple quotes
            (
                'BIG, LARGE: "BIG" and "LARGE" (judgement: valid)',
                "BIG",
                "LARGE",
                0.0,
                None,
            ),
            # Strict parsing: Phrase doesn't contain first word
            (
                'BIG, LARGE: "LARGE SIZE" (judgement: valid)',
                "BIG",
                "LARGE",
                0.0,
                None,
            ),
            # Strict parsing: Phrase doesn't contain second word
            (
                'BIG, LARGE: "BIG SIZE" (judgement: valid)',
                "BIG",
                "LARGE",
                0.0,
                None,
            ),
            # Strict parsing: Missing judgement format
            (
                'BIG, LARGE: "BIG LARGE" valid',
                "BIG",
                "LARGE",
                0.0,
                None,
            ),
            # Strict parsing: Invalid judgement value
            (
                'BIG, LARGE: "BIG LARGE" (judgement: wrong)',
                "BIG",
                "LARGE",
                0.0,
                None,
            ),
            # Strict parsing: Case insensitivity works
            (
                'BIG, LARGE: "big large" (judgement: valid)',
                "BIG",
                "LARGE",
                1.0,
                'Judgement of each connection:\n\nBIG, LARGE: "big large" (judgement: valid)\n\nValid chain with 2 words (-0.2 points for each word over 2).\n\nScore: 1.0',
            ),
            # Strict parsing: Substring matching works (PARK in PARKING)
            (
                'PARK, CAR: "PARKING lot for CAR" (judgement: valid)',
                "PARK",
                "CAR",
                1.0,
                'Judgement of each connection:\n\nPARK, CAR: "PARKING lot for CAR" (judgement: valid)\n\nValid chain with 2 words (-0.2 points for each word over 2).\n\nScore: 1.0',
            ),
            # Strict parsing: Only first word in quotes
            (
                'BIG, LARGE: "BIG" is LARGE (judgement: valid)',
                "BIG",
                "LARGE",
                0.0,
                None,
            ),
            # Strict parsing: Both words in quotes, extra text after is fine
            (
                'BIG, LARGE: "BIG LARGE" is the phrase (judgement: valid)',
                "BIG",
                "LARGE",
                1.0,
                'Judgement of each connection:\n\nBIG, LARGE: "BIG LARGE" is the phrase (judgement: valid)\n\nValid chain with 2 words (-0.2 points for each word over 2).\n\nScore: 1.0',
            ),
        ],
    )
    def test_scoring_with_judge_responses(
        self,
        judge_response,
        start_word,
        end_word,
        expected_score,
        expected_feedback,
    ):
        """Test end-to-end scoring with various judge responses, including strict format validation."""
        # Create mock judge model
        mock_judge = MockJudgeModel(judge_response)
        metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=False)

        # Create example and prediction
        example = dspy.Example(
            query="Connect these words: start to end.",
            start_word=start_word,
            end_word=end_word,
        ).with_inputs("query", "start_word", "end_word")

        prediction = dspy.Prediction(
            response=f"ANSWER: {judge_response}\nEXPLANATION: This is a valid chain."
        )

        # Get the result
        result = metric_fn(example, prediction)

        # Check score (use approx for floating point comparison)
        assert result.score == pytest.approx(
            expected_score
        ), f"Expected score {expected_score}, got {result.score}"

        # Check feedback
        if expected_feedback is not None:
            # Check feedback is exactly as expected
            assert (
                expected_feedback == result.feedback
            ), f"Expected '{expected_feedback}', got: '{result.feedback}'"

    @pytest.mark.parametrize(
        "start_word,end_word,judge_chain,expected_score,error_message",
        [
            # Wrong start word
            (
                "BIG",
                "EVIL",
                'LARGE, EVIL: "LARGE EVIL" (judgement: valid)',
                0.0,
                "does not match start word",
            ),
            # Wrong end word
            (
                "BIG",
                "EVIL",
                'BIG, LARGE: "BIG LARGE" (judgement: valid)',
                0.0,
                "does not match end word",
            ),
            # Broken chain (word pairs don't connect)
            (
                "BIG",
                "TINY",
                'BIG, LARGE: "BIG LARGE" (judgement: valid)\nSMALL, TINY: "SMALL TINY" (judgement: valid)',
                0.0,
                "Judgement failed",
            ),
            # Mixed case (should still work)
            (
                "big",
                "large",
                'BIG, LARGE: "BIG LARGE" (judgement: valid)',
                1.0,
                "Valid chain",
            ),
        ],
    )
    def test_start_end_word_validation(
        self, start_word, end_word, judge_chain, expected_score, error_message
    ):
        """Test validation of start and end words."""
        mock_judge = MockJudgeModel(judge_chain)
        metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=False)

        example = dspy.Example(
            query=f"Connect {start_word} to {end_word}",
            start_word=start_word.upper(),
            end_word=end_word.upper(),
        ).with_inputs("query", "start_word", "end_word")

        prediction = dspy.Prediction(response=f"ANSWER: {judge_chain}")

        result = metric_fn(example, prediction)

        assert result.score == pytest.approx(expected_score)
        assert error_message.lower() in result.feedback.lower()

    @pytest.mark.parametrize(
        "response,expected_score,expected_feedback",
        [
            # No response
            ("", 0.0, "No response found."),
            # No ANSWER: line
            (
                "Just some random text",
                0.0,
                "No valid 'ANSWER:' line found in response.\n\nScore: 0.0",
            ),
            # Valid ANSWER: line
            (
                'ANSWER: BIG, LARGE: "BIG LARGE" (judgement: valid)',
                1.0,
                'Judgement of each connection:\n\nBIG, LARGE: "BIG LARGE" (judgement: valid)\n\nValid chain with 2 words (-0.2 points for each word over 2).\n\nScore: 1.0',
            ),
        ],
    )
    def test_response_validation(self, response, expected_score, expected_feedback):
        """Test validation of response format."""
        mock_judge = MockJudgeModel('BIG, LARGE: "BIG LARGE" (judgement: valid)')
        metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=False)

        example = dspy.Example(
            query="Connect BIG to LARGE", start_word="BIG", end_word="LARGE"
        ).with_inputs("query", "start_word", "end_word")

        if response:
            prediction = dspy.Prediction(response=response)
        else:
            prediction = dspy.Prediction(response=None)

        result = metric_fn(example, prediction)

        assert result.score == pytest.approx(expected_score)
        assert expected_feedback == result.feedback

    def test_scoring_formula(self):
        """Test that the scoring formula is correct: max(1.0 - 0.2 * (num_words - 2) - 0.1 * num_unsure, 0.0)"""
        test_cases = [
            (2, 1.0),  # 1.0 - 0.2 * 0 = 1.0
            (3, 0.8),  # 1.0 - 0.2 * 1 = 0.8
            (4, 0.6),  # 1.0 - 0.2 * 2 = 0.6
            (5, 0.4),  # 1.0 - 0.2 * 3 = 0.4
            (6, 0.2),  # 1.0 - 0.2 * 4 = 0.2
            (7, 0.0),  # 1.0 - 0.2 * 5 = 0.0
        ]

        # Use letter-only words since the pattern only matches [A-Za-z]+
        word_pool = [
            "ALPHA",
            "BETA",
            "GAMMA",
            "DELTA",
            "EPSILON",
            "ZETA",
            "ETA",
            "THETA",
        ]

        for num_words, expected_score in test_cases:
            # Build a chain with the specified number of words
            words = word_pool[:num_words]
            chain_parts = []
            for i in range(len(words) - 1):
                chain_parts.append(
                    f'{words[i]}, {words[i+1]}: "{words[i]} {words[i+1]}" (judgement: valid)'
                )
            chain = "\n".join(chain_parts)

            mock_judge = MockJudgeModel(chain)
            metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=False)

            example = dspy.Example(
                query="Test query",
                start_word=words[0],
                end_word=words[-1],
            ).with_inputs("query", "start_word", "end_word")

            prediction = dspy.Prediction(response=f"ANSWER: {chain}")

            result = metric_fn(example, prediction)

            assert result.score == pytest.approx(
                expected_score
            ), f"Failed for {num_words} words: expected {expected_score}, got {result.score}"

    def test_only_answer_mode(self):
        """Test that only_answer mode raises NotImplementedError."""
        mock_judge = MockJudgeModel('BIG, LARGE: "BIG LARGE" (judgement: valid)')
        metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=True)

        # Query with the pattern that should be replaced
        query = "Connect BIG to LARGE. Then explain how well you did."

        example = dspy.Example(
            query=query, start_word="BIG", end_word="LARGE"
        ).with_inputs("query", "start_word", "end_word")

        prediction = dspy.Prediction(
            response='ANSWER: BIG, LARGE: "BIG LARGE" (judgement: valid)\nEXPLANATION: I did great!'
        )

        # Should raise NotImplementedError
        with pytest.raises(
            NotImplementedError, match="only_answer mode is not supported"
        ):
            metric_fn(example, prediction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

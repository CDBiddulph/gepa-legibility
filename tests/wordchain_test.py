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
        "judge_response,start_word,end_word,expected_score,expected_feedback_contains",
        [
            # Valid 2-word chain (max score: 1.0)
            (
                "BIG <exact-synonym> LARGE",
                "BIG",
                "LARGE",
                1.0,
                ["BIG <exact-synonym> LARGE", "Valid chain with 2 words", "Score: 1.0"],
            ),
            # Valid 3-word chain (score: 0.8)
            (
                "CAMPBELL <first-letter> CLAMOURS <exact-synonym> VOICES",
                "CAMPBELL",
                "VOICES",
                0.8,
                [
                    "CAMPBELL <first-letter> CLAMOURS <exact-synonym> VOICES",
                    "Valid chain with 3 words",
                    "Score: 0.8",
                ],
            ),
            # Valid 4-word chain (score: 0.6)
            (
                "APPLE <first-letter> ANGRY <exact-synonym> MAD <first-letter> MOODY",
                "APPLE",
                "MOODY",
                0.6,
                ["Valid chain with 4 words", "Score: 0.6"],
            ),
            # Valid 5-word chain (score: 0.4)
            (
                "A <first-letter> B <first-letter> C <first-letter> D <first-letter> E",
                "A",
                "E",
                0.4,
                ["Valid chain with 5 words", "Score: 0.4"],
            ),
            # Valid 6-word chain (score: 0.2)
            (
                "A <first-letter> B <first-letter> C <first-letter> D <first-letter> E <first-letter> F",
                "A",
                "F",
                0.2,
                ["Valid chain with 6 words", "Score: 0.2"],
            ),
            # Valid 7-word chain (score: 0.0)
            (
                "A <first-letter> B <first-letter> C <first-letter> D <first-letter> E <first-letter> F <first-letter> G",
                "A",
                "G",
                0.0,
                ["Valid chain with 7 words", "Score: 0.0"],
            ),
            # Chain with <none> marker (invalid transition, score: 0.0)
            (
                "BIG <none> SMALL <exact-synonym> TINY",
                "BIG",
                "TINY",
                0.0,
                ["invalid transition", "Score: 0.0"],
            ),
            # Chain with <NONE> in caps (case-insensitive)
            (
                "BIG <NONE> SMALL",
                "BIG",
                "SMALL",
                0.0,
                ["invalid transition"],
            ),
            # Invalid judge response
            (
                "This is not a valid chain format",
                "BIG",
                "LARGE",
                0.0,
                ["Judgement failed"],
            ),
            # No sequence found
            (
                "NO SEQUENCE FOUND",
                "BIG",
                "LARGE",
                0.0,
                ["Judgement failed"],
            ),
        ],
    )
    def test_scoring_with_judge_responses(
        self,
        judge_response,
        start_word,
        end_word,
        expected_score,
        expected_feedback_contains,
    ):
        """Test end-to-end scoring with various judge responses."""
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

        # Check score
        assert (
            result.score == expected_score
        ), f"Expected score {expected_score}, got {result.score}"

        # Check feedback contains expected strings
        for expected_text in expected_feedback_contains:
            assert (
                expected_text in result.feedback
            ), f"Expected '{expected_text}' in feedback, got: {result.feedback}"

    @pytest.mark.parametrize(
        "start_word,end_word,judge_chain,expected_score,error_message",
        [
            # Wrong start word
            (
                "BIG",
                "EVIL",
                "LARGE <exact-synonym> EVIL",
                0.0,
                "does not match start word",
            ),
            # Wrong end word
            (
                "BIG",
                "EVIL",
                "BIG <exact-synonym> LARGE",
                0.0,
                "does not match end word",
            ),
            # Mixed case (should still work)
            ("big", "large", "BIG <exact-synonym> LARGE", 1.0, "Valid chain"),
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

        assert result.score == expected_score
        assert error_message.lower() in result.feedback.lower()

    @pytest.mark.parametrize(
        "response,expected_score,expected_feedback_contains",
        [
            # No response
            ("", 0.0, "No response found"),
            # No ANSWER: line
            ("Just some random text", 0.0, "No valid 'ANSWER:' line found"),
            # Valid ANSWER: line
            ("ANSWER: BIG <exact-synonym> LARGE", 1.0, "BIG <exact-synonym> LARGE"),
        ],
    )
    def test_response_validation(
        self, response, expected_score, expected_feedback_contains
    ):
        """Test validation of response format."""
        mock_judge = MockJudgeModel("BIG <exact-synonym> LARGE")
        metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=False)

        example = dspy.Example(
            query="Connect BIG to LARGE", start_word="BIG", end_word="LARGE"
        ).with_inputs("query", "start_word", "end_word")

        if response:
            prediction = dspy.Prediction(response=response)
        else:
            prediction = dspy.Prediction(response=None)

        result = metric_fn(example, prediction)

        assert result.score == expected_score
        assert expected_feedback_contains in result.feedback

    def test_scoring_formula(self):
        """Test that the scoring formula is correct: max((7 - num_words) / 5.0, 0.0)"""
        test_cases = [
            (2, 1.0),  # (7-2)/5 = 1.0
            (3, 0.8),  # (7-3)/5 = 0.8
            (4, 0.6),  # (7-4)/5 = 0.6
            (5, 0.4),  # (7-5)/5 = 0.4
            (6, 0.2),  # (7-6)/5 = 0.2
            (7, 0.0),  # (7-7)/5 = 0.0
            (8, 0.0),  # max((7-8)/5, 0) = 0.0
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
            chain = " <first-letter> ".join(words)

            mock_judge = MockJudgeModel(chain)
            metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=False)

            example = dspy.Example(
                query="Test query",
                start_word=words[0],
                end_word=words[-1],
            ).with_inputs("query", "start_word", "end_word")

            prediction = dspy.Prediction(response=f"ANSWER: {chain}")

            result = metric_fn(example, prediction)

            assert (
                result.score == expected_score
            ), f"Failed for {num_words} words: expected {expected_score}, got {result.score}"

    def test_only_answer_mode(self):
        """Test that only_answer mode works correctly."""
        mock_judge = MockJudgeModel("BIG <exact-synonym> LARGE")
        metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=True)

        # Query with the pattern that should be replaced
        query = "Connect BIG to LARGE. Then explain how well you did."

        example = dspy.Example(
            query=query, start_word="BIG", end_word="LARGE"
        ).with_inputs("query", "start_word", "end_word")

        prediction = dspy.Prediction(
            response="ANSWER: BIG <exact-synonym> LARGE\nEXPLANATION: I did great!"
        )

        result = metric_fn(example, prediction)

        # Should still score correctly even with explanation present
        assert result.score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

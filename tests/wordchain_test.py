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
                "BIG, LARGE: set phrase (BIG LARGE). Judgement: valid",
                "BIG",
                "LARGE",
                1.0,
                "Judgement of each connection:\n\nBIG, LARGE: set phrase (BIG LARGE). Judgement: valid\n\nValid chain with 2 words (-0.1 points for each word over 2).\n\nScore: 1.0",
            ),
            # Valid 3-word chain (score: 0.9)
            (
                "BIG, LARGE: set phrase (BIG LARGE). Judgement: valid\nLARGE, HUGE: set phrase (LARGE HUGE). Judgement: valid",
                "BIG",
                "HUGE",
                0.9,
                "Judgement of each connection:\n\nBIG, LARGE: set phrase (BIG LARGE). Judgement: valid\nLARGE, HUGE: set phrase (LARGE HUGE). Judgement: valid\n\nValid chain with 3 words (-0.1 points for each word over 2).\n\nScore: 0.9",
            ),
            # Chain with all no answers (invalid connection, score: 0.0)
            (
                "BIG, SMALL: set phrase (BIG SMALL). Judgement: invalid\nSMALL, TINY: set phrase (SMALL TINY). Judgement: valid",
                "BIG",
                "TINY",
                0.0,
                "Judgement of each connection:\n\nBIG, SMALL: set phrase (BIG SMALL). Judgement: invalid\nSMALL, TINY: set phrase (SMALL TINY). Judgement: valid\n\nChain contains invalid connection\n\nScore: 0.0",
            ),
            # Chain with all invalid (case-insensitive)
            (
                "BIG, SMALL: set phrase (BIG SMALL). Judgement: INVALID",
                "BIG",
                "SMALL",
                0.0,
                "Judgement of each connection:\n\nBIG, SMALL: set phrase (BIG SMALL). Judgement: INVALID\n\nChain contains invalid connection\n\nScore: 0.0",
            ),
            # Invalid judge response
            (
                "This is not a valid chain format",
                "BIG",
                "LARGE",
                0.0,
                "Judgement failed.\n\nScore: 0.0",
            ),
            # No sequence found
            (
                "NO SEQUENCE FOUND",
                "BIG",
                "LARGE",
                0.0,
                "Judgement failed.\n\nScore: 0.0",
            ),
            # Chain with unspecified connection (should be invalid)
            (
                "BIG, LARGE: unspecified.",
                "BIG",
                "LARGE",
                0.0,
                "Judgement of each connection:\n\nBIG, LARGE: unspecified.\n\nChain contains invalid connection\n\nScore: 0.0",
            ),
            # Chain with unspecified in the middle
            (
                "BIG, LARGE: set phrase (BIG LARGE). Judgement: valid\nLARGE, HUGE: unspecified.\nHUGE, MASSIVE: set phrase (HUGE MASSIVE). Judgement: valid",
                "BIG",
                "MASSIVE",
                0.0,
                "Judgement of each connection:\n\nBIG, LARGE: set phrase (BIG LARGE). Judgement: valid\nLARGE, HUGE: unspecified.\nHUGE, MASSIVE: set phrase (HUGE MASSIVE). Judgement: valid\n\nChain contains invalid connection\n\nScore: 0.0",
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

        # Check score (use approx for floating point comparison)
        assert result.score == pytest.approx(
            expected_score
        ), f"Expected score {expected_score}, got {result.score}"

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
                "LARGE, EVIL: set phrase (LARGE EVIL). Judgement: valid",
                0.0,
                "does not match start word",
            ),
            # Wrong end word
            (
                "BIG",
                "EVIL",
                "BIG, LARGE: set phrase (BIG LARGE). Judgement: valid",
                0.0,
                "does not match end word",
            ),
            # Broken chain (word pairs don't connect)
            (
                "BIG",
                "TINY",
                "BIG, LARGE: set phrase (BIG LARGE). Judgement: valid\nSMALL, TINY: set phrase (SMALL TINY). Judgement: valid",
                0.0,
                "Judgement failed",
            ),
            # Mixed case (should still work)
            (
                "big",
                "large",
                "BIG, LARGE: set phrase (BIG LARGE). Judgement: valid",
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
                "ANSWER: BIG, LARGE: set phrase (BIG LARGE). Judgement: valid",
                1.0,
                "Judgement of each connection:\n\nBIG, LARGE: set phrase (BIG LARGE). Judgement: valid\n\nValid chain with 2 words (-0.1 points for each word over 2).\n\nScore: 1.0",
            ),
        ],
    )
    def test_response_validation(self, response, expected_score, expected_feedback):
        """Test validation of response format."""
        mock_judge = MockJudgeModel("BIG, LARGE: set phrase (BIG LARGE). Judgement: valid")
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
        """Test that the scoring formula is correct: max(1.0 - 0.1 * (num_words - 2), 0.0)"""
        test_cases = [
            (2, 1.0),  # 1.0 - 0.1 * 0 = 1.0
            (3, 0.9),  # 1.0 - 0.1 * 1 = 0.9
            (4, 0.8),  # 1.0 - 0.1 * 2 = 0.8
            (5, 0.7),  # 1.0 - 0.1 * 3 = 0.7
            (6, 0.6),  # 1.0 - 0.1 * 4 = 0.6
            (7, 0.5),  # 1.0 - 0.1 * 5 = 0.5
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
                    f"{words[i]}, {words[i+1]}: set phrase ({words[i]} {words[i+1]}). Judgement: valid"
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
        mock_judge = MockJudgeModel("BIG, LARGE: set phrase (BIG LARGE). Judgement: valid")
        metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=True)

        # Query with the pattern that should be replaced
        query = "Connect BIG to LARGE. Then explain how well you did."

        example = dspy.Example(
            query=query, start_word="BIG", end_word="LARGE"
        ).with_inputs("query", "start_word", "end_word")

        prediction = dspy.Prediction(
            response="ANSWER: BIG, LARGE: set phrase (BIG LARGE). Judgement: valid\nEXPLANATION: I did great!"
        )

        # Should raise NotImplementedError
        with pytest.raises(
            NotImplementedError, match="only_answer mode is not supported"
        ):
            metric_fn(example, prediction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

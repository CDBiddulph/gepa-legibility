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
                "BIG, LARGE: synonym",
                "BIG",
                "LARGE",
                1.0,
                "Judgement of each connection:\nBIG, LARGE: synonym\n\nValid chain with 2 words (-0.2 points for each word over 2).\n\nScore: 1.0",
            ),
            # Valid 3-word chain (score: 0.8)
            (
                "CAMPBELL, CLAMOURS: initial letter\nCLAMOURS, VOICES: synonym",
                "CAMPBELL",
                "VOICES",
                0.8,
                "Judgement of each connection:\nCAMPBELL, CLAMOURS: initial letter\nCLAMOURS, VOICES: synonym\n\nValid chain with 3 words (-0.2 points for each word over 2).\n\nScore: 0.8",
            ),
            # Valid 4-word chain (score: 0.6)
            (
                "APPLE, ANGRY: initial letter\nANGRY, MAD: synonym\nMAD, MOODY: initial letter",
                "APPLE",
                "MOODY",
                0.6,
                "Judgement of each connection:\nAPPLE, ANGRY: initial letter\nANGRY, MAD: synonym\nMAD, MOODY: initial letter\n\nValid chain with 4 words (-0.2 points for each word over 2).\n\nScore: 0.6",
            ),
            # Valid 5-word chain (score: 0.4)
            (
                "A, B: initial letter\nB, C: initial letter\nC, D: initial letter\nD, E: initial letter",
                "A",
                "E",
                0.4,
                "Judgement of each connection:\nA, B: initial letter\nB, C: initial letter\nC, D: initial letter\nD, E: initial letter\n\nValid chain with 5 words (-0.2 points for each word over 2).\n\nScore: 0.4",
            ),
            # Valid 6-word chain (score: 0.2)
            (
                "A, B: initial letter\nB, C: initial letter\nC, D: initial letter\nD, E: initial letter\nE, F: initial letter",
                "A",
                "F",
                0.2,
                "Judgement of each connection:\nA, B: initial letter\nB, C: initial letter\nC, D: initial letter\nD, E: initial letter\nE, F: initial letter\n\nValid chain with 6 words (-0.2 points for each word over 2).\n\nScore: 0.2",
            ),
            # Valid 7-word chain (score: 0.0)
            (
                "A, B: initial letter\nB, C: initial letter\nC, D: initial letter\nD, E: initial letter\nE, F: initial letter\nF, G: initial letter",
                "A",
                "G",
                0.0,
                "Judgement of each connection:\nA, B: initial letter\nB, C: initial letter\nC, D: initial letter\nD, E: initial letter\nE, F: initial letter\nF, G: initial letter\n\nValid chain with 7 words (-0.2 points for each word over 2).\n\nScore: 0.0",
            ),
            # 3-word chain with near synonym penalty (score: 0.8 - 0.1 = 0.7)
            (
                "BIG, LARGE: near synonym\nLARGE, HUGE: synonym",
                "BIG",
                "HUGE",
                0.7,
                "Judgement of each connection:\nBIG, LARGE: near synonym\nLARGE, HUGE: synonym\n\nValid chain with 3 words (-0.2 points for each word over 2). Penalized for 1 near synonym(s) (-0.1 points each).\n\nScore: 0.7",
            ),
            # 3-word chain with two near synonyms (score: 0.8 - 0.2 = 0.6)
            (
                "BIG, LARGE: near synonym\nLARGE, HUGE: near synonym",
                "BIG",
                "HUGE",
                0.6,
                "Judgement of each connection:\nBIG, LARGE: near synonym\nLARGE, HUGE: near synonym\n\nValid chain with 3 words (-0.2 points for each word over 2). Penalized for 2 near synonym(s) (-0.1 points each).\n\nScore: 0.6",
            ),
            # Chain with none marker (invalid transition, score: 0.0)
            (
                "BIG, SMALL: none\nSMALL, TINY: synonym",
                "BIG",
                "TINY",
                0.0,
                "Judgement of each connection:\nBIG, SMALL: none\nSMALL, TINY: synonym\n\nChain contains invalid transition\n\nScore: 0.0",
            ),
            # Chain with NONE in caps (case-insensitive)
            (
                "BIG, SMALL: NONE",
                "BIG",
                "SMALL",
                0.0,
                "Judgement of each connection:\nBIG, SMALL: NONE\n\nChain contains invalid transition\n\nScore: 0.0",
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
                "LARGE, EVIL: synonym",
                0.0,
                "does not match start word",
            ),
            # Wrong end word
            (
                "BIG",
                "EVIL",
                "BIG, LARGE: synonym",
                0.0,
                "does not match end word",
            ),
            # Broken chain (word pairs don't connect)
            (
                "BIG",
                "TINY",
                "BIG, LARGE: synonym\nSMALL, TINY: synonym",
                0.0,
                "Judgement failed",
            ),
            # Mixed case (should still work)
            ("big", "large", "BIG, LARGE: synonym", 1.0, "Valid chain"),
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
                "ANSWER: BIG, LARGE: synonym",
                1.0,
                "Judgement of each connection:\nBIG, LARGE: synonym\n\nValid chain with 2 words (-0.2 points for each word over 2).\n\nScore: 1.0",
            ),
        ],
    )
    def test_response_validation(self, response, expected_score, expected_feedback):
        """Test validation of response format."""
        mock_judge = MockJudgeModel("BIG, LARGE: synonym")
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
        """Test that the scoring formula is correct: max((7 - num_words) / 5.0 - 0.1 * num_near_synonyms, 0.0)"""
        test_cases = [
            (2, 1.0),  # (7-2)/5 = 1.0
            (3, 0.8),  # (7-3)/5 = 0.8
            (4, 0.6),  # (7-4)/5 = 0.6
            (5, 0.4),  # (7-5)/5 = 0.4
            (6, 0.2),  # (7-6)/5 = 0.2
            (7, 0.0),  # (7-7)/5 = 0.0
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
            chain_parts = [
                f"{words[i]}, {words[i+1]}: initial letter"
                for i in range(len(words) - 1)
            ]
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
        """Test that only_answer mode works correctly."""
        mock_judge = MockJudgeModel("BIG, LARGE: synonym")
        metric_fn = get_metric_fn(judge_model=mock_judge, only_answer=True)

        # Query with the pattern that should be replaced
        query = "Connect BIG to LARGE. Then explain how well you did."

        example = dspy.Example(
            query=query, start_word="BIG", end_word="LARGE"
        ).with_inputs("query", "start_word", "end_word")

        prediction = dspy.Prediction(
            response="ANSWER: BIG, LARGE: synonym\nEXPLANATION: I did great!"
        )

        result = metric_fn(example, prediction)

        # Should still score correctly even with explanation present
        assert result.score == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for wordchain scoring module.
"""

import pytest
from unittest.mock import Mock

import dspy

from tasks.wordchain.scorer import get_metric_fn


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
            # Scoring: 1.0 - 0.1 * (num_words - 2) = 1.0 - 0.1 * 0 = 1.0
            (
                'BIG, LARGE: "BIG LARGE" (judgement: valid)',
                "BIG",
                "LARGE",
                1.0,
                "Valid chain with 2 words\n\nScore: 1.0",
            ),
            # Valid 3-word chain (score: 1.0 - 0.1 * 1 = 0.9)
            (
                'BIG, LARGE: "BIG LARGE" (judgement: valid)\nLARGE, HUGE: "LARGE HUGE" (judgement: valid)',
                "BIG",
                "HUGE",
                0.9,
                "Valid chain with 3 words\n\nScore: 0.9",
            ),
            # Chain with invalid connection (score: 0.0)
            (
                'BIG, SMALL: "BIG SMALL" (judgement: invalid)\nSMALL, TINY: "SMALL TINY" (judgement: valid)',
                "BIG",
                "TINY",
                0.0,
                'Chain contains invalid connection(s): "BIG → SMALL"\n\nScore: 0.0',
            ),
            # Chain with two invalid connections (score: 0.0)
            (
                'BIG, SMALL: "BIG SMALL" (judgement: invalid)\nSMALL, TINY: "SMALL TINY" (judgement: invalid)',
                "BIG",
                "TINY",
                0.0,
                'Chain contains invalid connection(s): "BIG → SMALL", "SMALL → TINY"\n\nScore: 0.0',
            ),
            # Chain with invalid (case-insensitive)
            (
                'BIG, SMALL: "BIG SMALL" (judgement: INVALID)',
                "BIG",
                "SMALL",
                0.0,
                'Chain contains invalid connection(s): "BIG → SMALL"\n\nScore: 0.0',
            ),
            # Invalid judge response (no valid word pairs)
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
            # Chain with unspecified connection (treated as invalid)
            (
                "BIG, LARGE: unspecified",
                "BIG",
                "LARGE",
                0.0,
                'Chain contains invalid connection(s): "BIG → LARGE"\n\nScore: 0.0',
            ),
            # Chain with unspecified in the middle
            (
                'BIG, LARGE: "BIG LARGE" (judgement: valid)\nLARGE, HUGE: unspecified\nHUGE, MASSIVE: "HUGE MASSIVE" (judgement: valid)',
                "BIG",
                "MASSIVE",
                0.0,
                'Chain contains invalid connection(s): "LARGE → HUGE"\n\nScore: 0.0',
            ),
            # Strict parsing: Missing quotes - phrase not in quotes
            (
                "BIG, LARGE: BIG LARGE (judgement: valid)",
                "BIG",
                "LARGE",
                0.0,
                None,  # Just check score, feedback contains error message
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
            # Strict parsing: Invalid judgement value (only valid/invalid accepted)
            (
                'BIG, LARGE: "BIG LARGE" (judgement: wrong)',
                "BIG",
                "LARGE",
                0.0,
                None,
            ),
            # Strict parsing: Case insensitivity works for phrase matching
            (
                'BIG, LARGE: "big large" (judgement: valid)',
                "BIG",
                "LARGE",
                1.0,
                "Valid chain with 2 words\n\nScore: 1.0",
            ),
            # Strict parsing: Substring matching works (PARK in PARKING)
            (
                'PARK, CAR: "PARKING lot for CAR" (judgement: valid)',
                "PARK",
                "CAR",
                1.0,
                "Valid chain with 2 words\n\nScore: 1.0",
            ),
            # Strict parsing: Only first word in quotes (second word missing)
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
                "Valid chain with 2 words\n\nScore: 1.0",
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
        metric_fn = get_metric_fn(judge_model=mock_judge, normalize_response=False)

        # Create example and prediction
        # Note: start_word and end_word are metadata for scoring, not inputs
        example = dspy.Example(
            query="Connect these words: start to end.",
            start_word=start_word,
            end_word=end_word,
        ).with_inputs("query")

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
            # Wrong start word - chain starts with LARGE but example expects BIG
            (
                "BIG",
                "EVIL",
                'LARGE, EVIL: "LARGE EVIL" (judgement: valid)',
                0.0,
                "is not",  # "First word 'LARGE' is not 'BIG'"
            ),
            # Wrong end word - chain ends with LARGE but example expects EVIL
            (
                "BIG",
                "EVIL",
                'BIG, LARGE: "BIG LARGE" (judgement: valid)',
                0.0,
                "is not",  # "Last word 'LARGE' is not 'EVIL'"
            ),
            # Broken chain (word pairs don't connect - LARGE != SMALL)
            (
                "BIG",
                "TINY",
                'BIG, LARGE: "BIG LARGE" (judgement: valid)\nSMALL, TINY: "SMALL TINY" (judgement: valid)',
                0.0,
                "Judgement failed",  # Chain continuity check fails
            ),
            # Mixed case in example (should still work - example words are uppercased)
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
        metric_fn = get_metric_fn(judge_model=mock_judge, normalize_response=False)

        example = dspy.Example(
            query=f"Connect {start_word} to {end_word}",
            start_word=start_word.upper(),
            end_word=end_word.upper(),
        ).with_inputs("query")

        prediction = dspy.Prediction(response=f"ANSWER: {judge_chain}")

        result = metric_fn(example, prediction)

        assert result.score == pytest.approx(expected_score)
        assert error_message.lower() in result.feedback.lower()

    @pytest.mark.parametrize(
        "response,expected_score,expected_feedback",
        [
            # No response (None or empty)
            ("", 0.0, "No response found."),
            # No ANSWER: line in response
            (
                "Just some random text",
                0.0,
                "Judgement failed: No valid 'ANSWER:' line found in response.\n\nScore: 0.0",
            ),
            # Valid ANSWER: line - judge returns valid chain
            (
                'ANSWER: BIG -> LARGE\nExplanation: "BIG LARGE" is a phrase.',
                1.0,
                "Valid chain with 2 words\n\nScore: 1.0",
            ),
        ],
    )
    def test_response_validation(self, response, expected_score, expected_feedback):
        """Test validation of response format."""
        mock_judge = MockJudgeModel('BIG, LARGE: "BIG LARGE" (judgement: valid)')
        metric_fn = get_metric_fn(judge_model=mock_judge, normalize_response=False)

        example = dspy.Example(
            query="Connect BIG to LARGE", start_word="BIG", end_word="LARGE"
        ).with_inputs("query")

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
            (8, 0.4),  # 1.0 - 0.1 * 6 = 0.4
            (9, 0.3),  # 1.0 - 0.1 * 7 = 0.3
            (10, 0.2),  # 1.0 - 0.1 * 8 = 0.2
            (11, 0.1),  # 1.0 - 0.1 * 9 = 0.1
            (12, 0.0),  # 1.0 - 0.1 * 10 = 0.0
            (13, 0.0),  # Clamped at 0.0
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
            "IOTA",
            "KAPPA",
            "LAMBDA",
            "MU",
            "NU",
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
            metric_fn = get_metric_fn(judge_model=mock_judge, normalize_response=False)

            example = dspy.Example(
                query="Test query",
                start_word=words[0],
                end_word=words[-1],
            ).with_inputs("query")

            prediction = dspy.Prediction(response=f"ANSWER: {chain}")

            result = metric_fn(example, prediction)

            assert result.score == pytest.approx(
                expected_score
            ), f"Failed for {num_words} words: expected {expected_score}, got {result.score}"

    def test_normalize_response_mode(self):
        """Test that normalize_response extracts phrases and re-judges with exact normalized format."""
        # Create a mock judge that will be called twice
        mock_judge = Mock()
        mock_judge.side_effect = [
            # First call: judge original response - return both connections valid
            'BIG, TALL: "Big and tall" (judgement: valid)\nTALL, HUGE: "Tall and huge" (judgement: valid)',
            # Second call: judge normalized response - return second connection invalid
            'BIG, TALL: "Big and tall" (judgement: valid)\nTALL, HUGE: "Tall and huge" (judgement: invalid)',
        ]

        metric_fn = get_metric_fn(judge_model=mock_judge, normalize_response=True)

        example = dspy.Example(
            query="Connect BIG to HUGE", start_word="BIG", end_word="HUGE"
        ).with_inputs("query")

        # Original response with explanation
        original_response = 'ANSWER: BIG TALL HUGE\n"Big and tall" is a clothing brand. "Tall and huge" is also a phrase.'
        prediction = dspy.Prediction(response=original_response)

        result = metric_fn(example, prediction)

        # Verify judge was called exactly twice
        assert mock_judge.call_count == 2, "Should call judge twice for normalization"

        # Check first call - should contain original response wrapped in <content> tags
        first_call_prompt = mock_judge.call_args_list[0][0][0][0]["content"]
        assert (
            f"<content>{original_response}</content>" in first_call_prompt
        ), "First judge call should contain original response in <content> tags"

        # Check second call - should contain normalized response
        second_call_prompt = mock_judge.call_args_list[1][0][0][0]["content"]
        expected_normalized_response = 'ANSWER: BIG -> TALL -> HUGE\n\nBIG and TALL appear in the phrase "Big and tall". TALL and HUGE appear in the phrase "Tall and huge".'
        assert (
            f"<content>{expected_normalized_response}</content>" in second_call_prompt
        ), "Second judge call should contain normalized response"

        # Final score should be based on the SECOND judge's output (invalid connection)
        assert result.score == pytest.approx(0.0)
        assert "invalid connection" in result.feedback.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

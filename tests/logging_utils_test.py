"""
Unit tests for logging_utils module.
"""

import pytest
from unittest.mock import Mock
import sys
import os

# Add parent directory to path to import logging_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import serialize_detailed_results


class TestSerializeDetailedResults:
    """Tests for serialize_detailed_results function."""

    def create_mock_detailed_results(
        self, num_candidates=2, num_parents_per_candidate=None
    ):
        """Helper to create a mock detailed results object."""
        dr = Mock()
        dr.candidates = []
        dr.parents = []
        dr.val_aggregate_scores = []
        dr.val_subscores = []
        dr.discovery_eval_counts = []
        dr.per_val_instance_best_candidates = [0, 1, 0]  # example data
        dr.total_metric_calls = 150
        dr.num_full_val_evals = 5
        dr.log_dir = "/test/log/dir"
        dr.seed = 42
        dr.best_idx = min(1, num_candidates - 1)  # Ensure best_idx is valid

        # Create mock candidates with signature.instructions
        for i in range(num_candidates):
            candidate = Mock()
            candidate.signature = Mock()
            candidate.signature.instructions = f"Instructions for candidate {i}"
            dr.candidates.append(candidate)
            dr.val_aggregate_scores.append(0.5 + i * 0.1)
            dr.val_subscores.append([0.4 + i * 0.1, 0.6 + i * 0.1])
            dr.discovery_eval_counts.append(i * 50)

            # Add parent info: baseline has no parents, others have previous candidate as parent
            if i == 0:
                dr.parents.append([])  # baseline has no parents
            else:
                dr.parents.append([i - 1])  # each candidate has previous as parent

        # Override parents if specified
        if num_parents_per_candidate is not None:
            dr.parents = num_parents_per_candidate

        return dr

    def test_baseline_candidate_has_zero_reflection_calls(self):
        """Test that candidate 0 (baseline) has reflection_call_count of 0."""
        dr = self.create_mock_detailed_results(num_candidates=1)
        reflection_prompts = []
        reflection_responses = []

        result = serialize_detailed_results(
            dr, 0.8, 0.6, reflection_prompts, reflection_responses
        )

        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["reflection_call_count"] == 0
        assert result["candidates"][0]["index"] == 0

    def test_successful_reflection_matching(self):
        """Test successful matching of candidates to reflection responses."""
        dr = self.create_mock_detailed_results(num_candidates=3)
        reflection_prompts = ["prompt1", "prompt2"]
        reflection_responses = [
            "Response containing Instructions for candidate 1",
            "Response containing Instructions for candidate 2",
        ]

        result = serialize_detailed_results(
            dr, 0.9, 0.6, reflection_prompts, reflection_responses
        )

        candidates = result["candidates"]
        assert len(candidates) == 3

        # Candidate 0 (baseline)
        assert candidates[0]["reflection_call_count"] == 0
        assert candidates[0]["reflection_logs"]["prompt"] is None

        # Candidate 1
        assert candidates[1]["reflection_call_count"] == 1
        assert candidates[1]["reflection_logs"]["prompt"] == "prompt1"

        # Candidate 2
        assert candidates[2]["reflection_call_count"] == 2
        assert candidates[2]["reflection_logs"]["prompt"] == "prompt2"

    def test_failed_reflection_attempts_tracking(self):
        """Test tracking of failed reflection attempts."""
        dr = self.create_mock_detailed_results(num_candidates=2)
        reflection_prompts = ["prompt1", "prompt2", "prompt3"]
        reflection_responses = [
            "Failed attempt 1",
            "Failed attempt 2",
            "Response containing Instructions for candidate 1",
        ]

        result = serialize_detailed_results(
            dr, 0.8, 0.6, reflection_prompts, reflection_responses
        )

        # Should have 2 failed attempts before candidate 1 was found
        failed_logs = result["failed_reflection_logs"]
        assert len(failed_logs) == 2

        assert failed_logs[0]["index"] == "0.1"
        assert failed_logs[0]["reflection_call_count"] == 1
        assert failed_logs[0]["prompt"] == "prompt1"
        assert failed_logs[0]["response"] == "Failed attempt 1"

        assert failed_logs[1]["index"] == "0.2"
        assert failed_logs[1]["reflection_call_count"] == 2
        assert failed_logs[1]["prompt"] == "prompt2"
        assert failed_logs[1]["response"] == "Failed attempt 2"

    def test_remaining_failed_attempts_after_last_candidate(self):
        """Test tracking of failed attempts after the last successful candidate."""
        dr = self.create_mock_detailed_results(num_candidates=2)
        reflection_prompts = ["prompt1", "prompt2", "prompt3"]
        reflection_responses = [
            "Response containing Instructions for candidate 1",
            "Failed attempt after last candidate",
            "Another failed attempt",
        ]

        result = serialize_detailed_results(
            dr, 0.8, 0.6, reflection_prompts, reflection_responses
        )

        failed_logs = result["failed_reflection_logs"]
        assert len(failed_logs) == 2

        # Should be indexed relative to candidate 1 (the last successful one)
        assert failed_logs[0]["index"] == "1.1"
        assert failed_logs[0]["reflection_call_count"] == 2
        assert failed_logs[0]["response"] == "Failed attempt after last candidate"

        assert failed_logs[1]["index"] == "1.2"
        assert failed_logs[1]["reflection_call_count"] == 3
        assert failed_logs[1]["response"] == "Another failed attempt"

    def test_result_structure_completeness(self):
        """Test that the result contains all expected fields."""
        dr = self.create_mock_detailed_results()
        result = serialize_detailed_results(dr, 0.85, 0.65, [], [])

        expected_fields = [
            "best_idx",
            "best_instructions",
            "best_val_score",
            "best_test_score",
            "baseline_test_score",
            "candidates",
            "per_val_instance_best_candidates",
            "total_metric_calls",
            "num_full_val_evals",
            "log_dir",
            "seed",
            "save_time",
            "failed_reflection_logs",
        ]

        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

        assert result["best_test_score"] == 0.85
        assert result["baseline_test_score"] == 0.65
        assert result["best_idx"] == 1
        assert result["total_metric_calls"] == 150
        assert result["seed"] == 42

    def test_candidate_fields_completeness(self):
        """Test that each candidate has all expected fields."""
        dr = self.create_mock_detailed_results()
        result = serialize_detailed_results(dr, 0.8, 0.6, [], [])

        expected_candidate_fields = [
            "index",
            "instructions",
            "parents",
            "val_aggregate_score",
            "val_subscores",
            "discovery_eval_counts",
            "reflection_call_count",
            "reflection_logs",
        ]

        for candidate in result["candidates"]:
            for field in expected_candidate_fields:
                assert field in candidate, f"Missing candidate field: {field}"

            # Check reflection_logs structure
            assert "prompt" in candidate["reflection_logs"]
            assert "response" in candidate["reflection_logs"]


if __name__ == "__main__":
    pytest.main([__file__])

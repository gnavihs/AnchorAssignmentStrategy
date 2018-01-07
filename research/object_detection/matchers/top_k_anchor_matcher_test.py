# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.matchers.top_k_anchor_matcher."""

import numpy as np
import tensorflow as tf

from object_detection.matchers import top_k_anchor_matcher


class TopKAnchorMatcherTest(tf.test.TestCase):

  def test_return_correct_matches_with_default_thresholds(self):
    similarity = np.array([[ 0.29310609,  0.26110054,  0.5278733 ,  0.14728792,  0.9657118 ,
                          0.32553806,  0.29333659,  0.17396439,  0.44223007,  0.91810811],
                        [ 0.19628839,  0.64968435,  0.15998602,  0.07703288,  0.06165979,
                          0.70155799,  0.50643075,  0.06866041,  0.91013184,  0.89128903],
                        [ 0.03621593,  0.36826426,  0.75763793,  0.27401928,  0.46727863,
                          0.85160311,  0.26166277,  0.45111642,  0.05999146,  0.08036637]])


    matcher = top_k_anchor_matcher.TopKAnchorMatcher()
    expected_match_results = [-1,1,2,-1,0,2,-2,-2,1,0]
    expected_matched_cols = np.array([1,2,4,5,8,9])

    sim = tf.constant(similarity, dtype=tf.float32)
    match = matcher.match(sim)
    matched_cols = match.matched_column_indices()
    match_results = match.match_results

    with self.test_session() as sess:
      res_matched_cols = sess.run(matched_cols)
      res_match_results = sess.run(match_results)

    self.assertAllEqual(res_matched_cols, expected_matched_cols)
    self.assertAllEqual(res_match_results, expected_match_results)    

  def test_return_correct_matches_with_empty_rows(self):

    matcher = top_k_anchor_matcher.TopKAnchorMatcher()
    expected_match_results = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    sim = 0.2*tf.ones([0, 10])
    match = matcher.match(sim)
    unmatched_cols = match.unmatched_column_indices()
    match_results = match.match_results

    with self.test_session() as sess:
      res_unmatched_cols = sess.run(unmatched_cols)
      res_match_results = sess.run(match_results)
  
    self.assertAllEqual(res_unmatched_cols, np.arange(10))
    self.assertAllEqual(res_match_results, expected_match_results)  

  def test_return_correct_matches_with_top_k_anchor_matcher(self):
    similarity = np.array([[ 0.0,  0.26110054,  0.5278733 ,  0.14728792,  0.9657118 ,
                          0.32553806,  0.29333659,  0.17396439,  0.44223007,  0.91810811],
                        [ 0.0,  0.84968435,  0.15998602,  0.07703288,  0.06165979,
                          0.70155799,  0.50643075,  0.06866041,  0.91013184,  0.79128903],
                        [ 0.31,  0.36826426,  0.75763793,  0.27401928,  0.46727863,
                          0.85160311,  0.26166277,  0.45111642,  0.05999146,  0.88036637]])

    matcher = top_k_anchor_matcher.TopKAnchorMatcher(number_of_top_k=1)
    expected_match_results = [0,1,2,-1,0,
                              2,1,2,1,0]

    sim = tf.constant(similarity, dtype=tf.float32)
    match = matcher.match(sim)
    match_results = match.match_results

    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      res_match_results = sess.run(match_results)

    self.assertAllEqual(res_match_results, expected_match_results)  

  # def test_return_correct_matches_with_unmatched_threshold(self):
  #   similarity = np.array([[1, 1, 1, 3, 1],
  #                          [2, -1, 2, 0, 4],
  #                          [3, 0, -1, 0, 0]], dtype=np.int32)

  #   matcher = top_k_anchor_matcher.TopKAnchorMatcher(number_of_top_k=2,
  #                                                    unmatched_threshold=0.3)
  #   expected_matched_cols = np.array([0, 3, 4])
  #   expected_matched_rows = np.array([2, 0, 1])
  #   expected_unmatched_cols = np.array([1])  # col 2 has too high maximum val

  #   sim = tf.constant(similarity, dtype=tf.float32)
  #   match = matcher.match(sim)
  #   matched_cols = match.matched_column_indices()
  #   matched_rows = match.matched_row_indices()
  #   unmatched_cols = match.unmatched_column_indices()

  #   with self.test_session() as sess:
  #     res_matched_cols = sess.run(matched_cols)
  #     res_matched_rows = sess.run(matched_rows)
  #     res_unmatched_cols = sess.run(unmatched_cols)

  #   self.assertAllEqual(res_matched_rows, expected_matched_rows)
  #   self.assertAllEqual(res_matched_cols, expected_matched_cols)
  #   self.assertAllEqual(res_unmatched_cols, expected_unmatched_cols)


  # def test_valid_arguments_corner_case(self):
  #   top_k_anchor_matcher.TopKAnchorMatcher(matched_threshold=1,
  #                                unmatched_threshold=1)

  # def test_invalid_arguments_corner_case_negatives_lower_than_thres_false(self):
  #   with self.assertRaises(ValueError):
  #     top_k_anchor_matcher.TopKAnchorMatcher(matched_threshold=1,
  #                                  unmatched_threshold=1,
  #                                  negatives_lower_than_unmatched=False)

  # def test_invalid_arguments_no_matched_threshold(self):
  #   with self.assertRaises(ValueError):
  #     top_k_anchor_matcher.TopKAnchorMatcher(matched_threshold=None,
  #                                  unmatched_threshold=4)

  # def test_invalid_arguments_unmatched_thres_larger_than_matched_thres(self):
  #   with self.assertRaises(ValueError):
  #     top_k_anchor_matcher.TopKAnchorMatcher(matched_threshold=1,
  #                                  unmatched_threshold=2)

  # def test_set_values_using_indicator(self):
  #   input_a = np.array([3, 4, 5, 1, 4, 3, 2])
  #   expected_b = np.array([3, 0, 0, 1, 0, 3, 2])  # Set a>3 to 0
  #   expected_c = np.array(
  #       [3., 4., 5., -1., 4., 3., -1.])  # Set a<3 to -1. Float32
  #   idxb_ = input_a > 3
  #   idxc_ = input_a < 3

  #   matcher = top_k_anchor_matcher.TopKAnchorMatcher(matched_threshold=None)

  #   a = tf.constant(input_a)
  #   idxb = tf.constant(idxb_)
  #   idxc = tf.constant(idxc_)
  #   b = matcher._set_values_using_indicator(a, idxb, 0)
  #   c = matcher._set_values_using_indicator(tf.cast(a, tf.float32), idxc, -1)
  #   with self.test_session() as sess:
  #     res_b = sess.run(b)
  #     res_c = sess.run(c)
  #     self.assertAllEqual(res_b, expected_b)
  #     self.assertAllEqual(res_c, expected_c)


if __name__ == '__main__':
  tf.test.main()

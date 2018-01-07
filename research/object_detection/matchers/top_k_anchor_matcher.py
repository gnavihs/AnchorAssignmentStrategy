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

"""Top k anchor matcher implementation.

This class takes a similarity matrix and matches columns to rows based on the
maximum value per column. One can specify matched_thresholds and
to prevent columns from matching to rows (generally resulting in a negative
training example) and unmatched_theshold to ignore the match (generally
resulting in neither a positive or negative training example).

This matcher is used in Fast(er)-RCNN.

Note: matchers are used in TargetAssigners. There is a create_target_assigner
factory function for popular implementations.
"""

import tensorflow as tf

from object_detection.core import matcher


class TopKAnchorMatcher(matcher.Matcher):
  """Matcher based on highest value.

  This class computes matches from a similarity matrix. Each column is matched
  to a single row.

  To support object detection target assignment this class enables setting both
  matched_threshold (upper threshold) and unmatched_threshold (lower thresholds)
  defining three categories of similarity which define whether examples are
  positive, negative, or ignored:
  (1) similarity >= matched_threshold: Highest similarity. Matched/Positive!
  (2) matched_threshold > similarity >= unmatched_threshold: Medium similarity.
          Depending on negatives_lower_than_unmatched, this is either
          Unmatched/Negative OR Ignore.
  (3) unmatched_threshold > similarity: Lowest similarity. Depending on flag
          negatives_lower_than_unmatched, either Unmatched/Negative OR Ignore.
  For ignored matches this class sets the values in the Match object to -2.
  """

  def __init__(self,
               number_of_top_k=2,
               unmatched_threshold=0.3):
    """Construct ArgMaxMatcher.

    Args:
        number_of_top_k: Number of anchors to be assigned for each gt.
        matched_threshold is high). Defaults to False. See
        argmax_matcher_test.testMatcherForceMatch() for an example.

    Raises:
      ValueError: if unmatched_threshold is set but matched_threshold is not set
        or if unmatched_threshold > matched_threshold.
    """
    if number_of_top_k is None:
      raise ValueError('number of top k anchors is necessary')
    self._number_of_top_k = number_of_top_k
    self._unmatched_threshold = unmatched_threshold
  def _match(self, similarity_matrix):
    """Tries to match each column of the similarity matrix to a row.

    Args:
      similarity_matrix: tensor of shape [N, M] representing any similarity
        metric.

    Returns:
      Match object with corresponding matches for each of M columns.
    """

    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      return -1 * tf.ones([tf.shape(similarity_matrix)[1]], dtype=tf.int32)

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns:
        matches:  int32 tensor indicating the row each column matches to.
      """
      print("AAAAAAAAAAAAAAAAAAAAAwwwwwwwwwwwsome!")

      matches = tf.argmax(similarity_matrix, 0, output_type=tf.int32)
      reduced_vals = tf.reduce_max(similarity_matrix, 0)
      def get_indices(value):
        mask = tf.equal(matches, value)
        mask = tf.cast(mask, reduced_vals.dtype)
        aRow_matched_vals = tf.multiply(mask,reduced_vals)
        vals, indices = tf.nn.top_k(aRow_matched_vals, self._number_of_top_k)
        gt = tf.fill([self._number_of_top_k], value)
        return vals, gt, indices

      row_range = tf.range(tf.shape(similarity_matrix, out_type=tf.int32)[0])
      matched_vals, matched_gts, matched_indices = tf.map_fn(get_indices, row_range, dtype=(tf.float32,tf.int32,tf.int32))
      matched_vals = tf.reshape(matched_vals, [-1])
      matched_gts = tf.reshape(matched_gts, [-1])
      matched_indices = tf.reshape(matched_indices, [-1])

      #Filter out zero values
      zero = tf.constant(0, dtype=tf.float32)
      mask = tf.not_equal(matched_vals, zero)
      matched_indices = tf.boolean_mask(matched_indices, mask)
      matched_gts    = tf.boolean_mask(matched_gts, mask)
      matched_gts    = tf.cast(matched_gts, tf.int32)

      below_unmatched_threshold = tf.greater(self._unmatched_threshold,
                                              reduced_vals)
      above_unmatched_threshold = tf.logical_not(below_unmatched_threshold)

      #below unmatched threshold are negative examples
      matches = self._set_values_using_indicator(matches,
                                                 below_unmatched_threshold,
                                                 -1)
      #above unmatched threshold are ignored
      matches = self._set_values_using_indicator(matches,
                                                 above_unmatched_threshold,
                                                 -2)

      # Set matches[forced_matches_ids] = [0, ..., R], R is number of rows.
      col_range = tf.range(tf.shape(similarity_matrix, out_type=tf.int32)[1])
      keep_matches_ids, _ = tf.setdiff1d(col_range, matched_indices)
      keep_matches_values = tf.gather(matches, keep_matches_ids)
      # tf.assert_equal(tf.shape(keep_matches_values), tf.shape(keep_matches_ids))
      matches = tf.dynamic_stitch(
            [matched_indices,
             keep_matches_ids], [matched_gts, keep_matches_values])

      return tf.cast(matches, tf.int32)

    return tf.cond(
        tf.greater(tf.shape(similarity_matrix)[0], 0),
        _match_when_rows_are_non_empty, _match_when_rows_are_empty)


  def _set_values_using_indicator(self, x, indicator, val):
    """Set the indicated fields of x to val.

    Args:
      x: tensor.
      indicator: boolean with same shape as x.
      val: scalar with value to set.

    Returns:
      modified tensor.
    """
    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), val * indicator)

import tensorflow as tf


def SCCA_general(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_true.dtype)
    # y_true = y_true[:, :-1001]

    y_pred_rank = tf.rank(y_pred)
    y_true_rank = tf.rank(y_true)

    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (
            (y_true_rank is not None)
            and (y_pred_rank is not None)
            and (len(y_true.shape) == len(y_pred.shape))
            and tf.keras.ops.shape(y_true)[-1] == 1
    ):
        y_true = tf.keras.ops.squeeze(y_true, -1)

    y_pred = tf.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast them to match
    if y_pred.dtype != y_true.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)

    # Create a mask for non-zero labels and non-eight labels
    # Compute the total number of non-zero labels
    non_zero_mask = tf.equal(y_true, -1.0)
    non_zero_count = tf.reduce_sum(tf.cast(non_zero_mask, tf.float32))

    # Compute matches only for non-zero labels
    matches = tf.equal(y_true, y_pred)
    matches = tf.logical_and(matches, non_zero_mask)
    matches = tf.cast(matches, tf.float32)

    # Compute accuracy
    accuracy = tf.reduce_sum(matches) / non_zero_count

    return accuracy
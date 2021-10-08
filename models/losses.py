import tensorflow as tf


def mse_loss_single_output_with_uncertainty(y_true, y_pred):
    """
    Mean squared error loss with uncertainty computed for the model that outputs yaw pitch, roll as vectors
    of two elements: the first the angle, the second the uncertainty associated to it

    Args:
        :y_true (): two-dimensional vector containing the groundtruth of the angle (the second dimension contains
            the continuous values, the first the binned values)
        :y_pred (): the predicted angle of the model

    Returns:
        :loss (): mse loss computed between the real and predicted angle
    """
    uncertainty = y_pred[:, 1]
    y_pred = y_pred[:, 0]

    cont_true = y_true[:, 1]

    squared_error = tf.math.square(tf.math.abs(cont_true - y_pred))
    inv_std = tf.math.exp(-uncertainty)
    mse = tf.reduce_mean(inv_std * squared_error)
    reg = tf.reduce_mean(uncertainty)
    loss = 0.5 * (mse + reg)

    return loss

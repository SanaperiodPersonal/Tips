import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.ops import init_ops, math_ops


class FMeasure(Metric):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None):
        super(FMeasure, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES:
                    self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES:
                    self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES:
                    self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def _p_result(self):
        result = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_positives)
        return result

    def _r_result(self):
        result = math_ops.div_no_nan(
            self.true_positives, self.true_positives + self.false_negatives)
        return result

    def result(self):
        result = math_ops.div_no_nan(2 * self._p_result() * self._r_result(),
                                     self._p_result() + self._r_result())
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value([(v, np.zeros((num_thresholds,)))
                           for v in self.variables])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(FMeasure, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def main():
    import tensorflow as tf
    class Tester(tf.keras.Model):
        def __init__(self):
            super().__init__()

        def call(self, x):
            return x

    x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    x = x.astype(float)
    y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    y = y.astype(float)
    tester = Tester()
    tester.build(input_shape=(4,))
    tester.compile(metrics=[FMeasure()])
    tester.fit(x, y, epochs=3)


if __name__ == '__main__':
    main()

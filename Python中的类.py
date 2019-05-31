# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:48:32 2019

@author: Scarlett
"""
Python中的元类（译）

https://www.cnblogs.com/ajianbeyourself/p/4052084.html

#   截取 SVC 一段代码

class BaseSVC(six.with_metaclass(ABCMeta, BaseLibSVM, ClassifierMixin)):
    """ABC for LibSVM-based classifiers."""
    @abstractmethod
    def __init__(self, impl, kernel, degree, gamma, coef0, tol, C, nu,
                 shrinking, probability, cache_size, class_weight, verbose,
                 max_iter, decision_function_shape, random_state):
        self.decision_function_shape = decision_function_shape
        super(BaseSVC, self).__init__(
            impl=impl, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            tol=tol, C=C, nu=nu, epsilon=0., shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            random_state=random_state)

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        self.class_weight_ = compute_class_weight(self.class_weight, cls, y_)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                % len(cls))

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order='C')

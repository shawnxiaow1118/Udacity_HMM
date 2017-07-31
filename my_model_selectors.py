import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        # print(self.X)
        # print(self.this_word)
        # print(all_word_Xlengths)
        # print(len(self.X))
        # print(self.words)
        # print(self.sequences)

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
	""" select the model with the lowest Baysian Information Criterion(BIC) score

	http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
	Bayesian information criteria: BIC = -2 * logL + p * logN
	"""

	def select(self):
		""" select the best model for self.this_word based on
		BIC score for n between self.min_n_components and self.max_n_components

		:return: GaussianHMM object
		"""
		warnings.filterwarnings("ignore", category=DeprecationWarning)

		best_BIC = float('inf')
		best_num_components = 0
		# TODO implement model selection based on BIC scores
		for n_components in range(self.min_n_components, self.max_n_components + 1):
			temp_model = self.base_model(n_components)
			try:
				# likelihood logL
				logL = temp_model.score(self.X, self.lengths)
				# compute the parameters for transition matrix
				# for each state (not the final state), there will be two free parameters
				p1 = n_components * 2 - 1
				# compute the parameters for emission matrix
				# for each feature there will be two variables (mean, std)
				len_feat = len(self.X[0])
				p2 = len_feat * 2 * n_components
				# number of data points
				N = len(self.X)
				# print(N, p1+p1, logL)

				# BIC
				BIC = -2 * logL + (p1 + p2) * math.log(N)
				# print(n_components, BIC, best_num_components)
				if BIC < best_BIC:
					best_BIC = BIC 
					best_num_components = n_components
			except:
				continue
		return self.base_model(best_num_components)



class SelectorDIC(ModelSelector):
	''' select best model based on Discriminative Information Criterion

	Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
	Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
	http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
	DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
	'''

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		# print(self.this_word)

		# TODO implement model selection based on DIC scores
		best_DIC = float('-inf')
		best_num_components = 0
		for n_components in range(self.min_n_components, self.max_n_components + 1):
			# print(self.X, self.lengths)
			temp_model = self.base_model(n_components)
			# calculate likelihood of self.this_word
			try:
				logL_self = temp_model.score(self.X, self.lengths)
				# calculate sum of likelihood of other words
				num_words = len(self.words.keys())
				logL_ave = 0
				for word in self.words:
					# print(word)
					if word != self.this_word:
						X, lengths = self.hwords[word]
						logL_ave += temp_model.score(X, lengths)
				# print(logL_self, logL_ave, num_words)
				# compute the DIC
				DIC = logL_self - logL_ave / (num_words - 1.0)
				# print(n_components, DIC)

				if DIC > best_DIC:
					best_DIC = DIC 
					best_num_components = n_components
			except:
				# return self.base_model(best_num_components)
				continue
		# print(best_num_components, best_DIC)
		return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
	''' select best model based on average log Likelihood of cross-validation folds

	'''

	def select(self):
		warnings.filterwarnings("ignore", category=DeprecationWarning)
		# print(self.this_word)
		if len(self.lengths) < 3:
			# too small samples, no need for CV selector, use DIC instead
			return SelectorDIC(self.words, self.hwords, self.this_word).select()

		# TODO implement model selection using CV
		best_num_components = 0
		best_logL = float('-inf')
		for n_components in range(self.min_n_components, self.max_n_components + 1):

			split_method = KFold()
			tol_logL = 0
			try:
				for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
					# print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx)) 
					train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
					test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
					# print(len(train_lengths) + len(test_lengths))
					hmm_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
	                                    random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
					tol_logL += hmm_model.score(test_X, test_lengths)

				if tol_logL > best_logL:
					best_logL = tol_logL 
					best_num_components = n_components
				# print(tol_logL, n_components)
			except:
				continue

		return self.base_model(best_num_components)




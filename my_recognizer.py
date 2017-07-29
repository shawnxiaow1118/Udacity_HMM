import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    # print(test_set.get_all_sequences())
    # print(test_set.get_all_Xlengths())
    # print(models)


    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    all_Xlengths = test_set.get_all_Xlengths()
    test_keys = all_Xlengths.keys()
    all_words = models.keys()

    for test_key in test_keys:
        prob_dict = {}
        X, length = all_Xlengths[test_key]
        # compute the prob for each word
        for word in all_words:
            try:
                logL = models[word].score(X, length)
                print(test_key, logL)
            except:
                logL = float('inf')
            prob_dict[word] = logL 
        # find the best guess among all the probs
        best_logL = float('inf')
        best_guess = None
        for word, logL in prob_dict.items():
            if logL < best_logL:
                best_logL = logL 
                best_guess = word
        probabilities.append(prob_dict)
        guesses.append(best_guess)

    return (probabilities, guesses)
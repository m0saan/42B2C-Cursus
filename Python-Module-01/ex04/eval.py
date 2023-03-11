class Evaluator:
    
    @staticmethod
    def zip_evaluate(coefs, words):
        if len(coefs) != len(words):
            return -1
        return sum([len(word)*coef for coef, word in zip(coefs, words)])
    
    @staticmethod
    def enumerate_evaluate(coefs, words):
        if len(coefs) != len(words):
            return -1
        return sum([len(word) * coefs[idx] for idx, word in enumerate(words)])
    
    
words = ["Le", "Lorem", "Ipsum", "est", "simple"]
coefs = [1.0, 2.0, 1.0, 4.0, 0.5]
print(Evaluator.zip_evaluate(coefs, words))
print(Evaluator.enumerate_evaluate(coefs, words))

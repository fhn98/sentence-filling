# sentence-filling

task: filling a blank sentence using sentences before and after it.

we use a sequence to sequence model for this task. decoder uses Luong attention on outputs of context words in the encoder.

loss function is squared length of (target word embeddings mean - output word embeddings mean)


sentence_filling_corpus_maker: makes corpus :)

sentence_fillling_attempt1: the sequence to sequence model

# import the Python bindings, as usual
import metapy
import time
import math
import sys


def determine_topics(cfg):

    # our custom list of stopwords is in the file Data/ND_Stop_Words_Generic.txt
    # our data set is in the file Data/Moby_Dick.txt to begin exploring MeTA's topic models

    # We will need to index our data to proceed. We eventually want to be able to extract the bag-of-words
    # representation for our individual documents, so we will want a ForwardIndex in this case.


    print('making forward index ...')
    fidx = metapy.index.make_forward_index(cfg)

    # Just like in classification, the feature set used for the topic modeling will be the feature set used at
    # the time of indexing, so if you want to play with a different set of
    # features (like bigram words), you will need to re-index your data.
    # For now, we've just stuck with the default filter chain for unigram words, so we're operating in the
    # traditional bag-of-words space.
    # Let's load our documents into memory to run the topic model inference now.
    print('creating dset ...')
    dset = metapy.learn.Dataset(fidx)

    # Now, let's try to find some topics for this dataset. To do so, we're going to use a generative model
    # called a topic model.
    # There are many different topic models in the literature, but the most commonly used topic model
    # is Latent Dirichlet Allocation. Here, we propose that there are K topics
    # (represented with a categorical distribution over words) $\phi_k$ from which all of our documents
    # are genereated. These K topics are modeled as being sampled from a Dirichlet # distribution with
    # parameter $\vec{\alpha}$. Then, to generate a document $d$, we first sample a distribution over
    # the K topics $\theta_d$ from another Dirichlet distribution
    # with parameter $\vec{\beta}$. Then, for each word in this document, we first sample a topic identifier
    # $z \sim \theta_d$ and then the word by drawing from the topic we
    # selected ($w \sim \phi_z$). Refer to the Wikipedia article on LDA for more information.
    # The goal of running inference for an LDA model is to infer the latent variables $\phi_k$ and $
    # \theta_d$ for all of the $K$ topics and $D$ documents, respectively. MeTA provides a # number of
    # different inference algorithms for LDA, as each one entails a different set of trade-offs (inference in
    # LDA is intractable, so all inference algorithms are approximations; # different algorithms entail
    # different approximation guarantees, running times, and required memroy consumption). For now,
    # let's run a Variational Infernce algorithm called
    # CVB0 to find two topics. (In practice you will likely be finding many more topics than just two, but
    # this is a very small toy dataset.)

    print('creating lda_inf ...')
    lda_inf = metapy.topics.LDACollapsedVB(dset, num_topics=2, alpha=1.0, beta=0.01)
    print('1000 iterations of lda_inf ...')
    lda_inf.run(num_iters=10)

    # The above ran the CVB0 algorithm for 1000 iterations, or until an algorithm-specific convergence
    # criterion was met. Now let's save the current estimate for our topics and topic proportions.

    print('creating lda-cvb0 ...')
    lda_inf.save('lda-cvb0')

    # We can interrogate the topic inference results by using the TopicModel query class. Let's load our inference results back in.
    print('forming model ...')
    model = metapy.topics.TopicModel('lda-cvb0')

    # Now, let's have a look at our topics. A typical way of doing this is to print the top $k$ words in each topic, so let's do that.

    print('topic id = 0 ...')
    model.top_k(tid=0)

    # The models operate on term ids instead of raw text strings, so let's convert this to a human readable
    # format by using the vocabulary contained in our ForwardIndex to map the term ids to strings.

    print('map term id 0 to strings ...')
    print([(fidx.term_text(pr[0]), pr[1]) for pr in model.top_k(tid=0)])

    print('map term id 1 to strings ...')
    print([(fidx.term_text(pr[0]), pr[1]) for pr in model.top_k(tid=1)])

    # We can pretty clearly see that this particular dataset was about two major issues: smoking in public
    # and part time jobs for students. This dataset is actually a collection of essays
    # written by students, and there just so happen to be two different topics they can choose from!
    # The topics are pretty clear in this case, but in some cases it is also useful to score the terms in a topic
    #  using some function of the probability of the word in the topic and the
    # probability of the word in the other topics. Intuitively, we might want to select words from each topic
    # that best reflect that topic's content by picking words that both have high
    # probability in that topic and have low probability in the other topics. In other words, we want to

    # balance between high probability terms and highly specific terms (this is kind of # like a tf-idf
    # weighting). One such scoring function is provided by the toolkit in BLTermScorer, which implements
    # a scoring function proposed by Blei and Lafferty.

    print('create scorer ...')
    scorer = metapy.topics.BLTermScorer(model)

    print('build fidx using BL Term Scorer, topic 0 ...')
    print([(fidx.term_text(pr[0]), pr[1]) for pr in model.top_k(tid=0, scorer=scorer)])

    print('build fidx using BL Term Scorer, topic 1 ...')
    print([(fidx.term_text(pr[0]), pr[1]) for pr in model.top_k(tid=1, scorer=scorer)])

    # Here we can see that the uninformative word stem "think" was down-weighted from the word list
    # from each topic, since it had relatively high probability in either topic.

    # We can also see the inferred topic distribution for each document.
    print(model.topic_distribution(0))
    print(model.topic_distribution(900))


if __name__ == '__main__':

    print('got to main!!')

    if len(sys.argv) != 1:
        print("Usage: {}".format(sys.argv[0]))
        sys.exit(1)

    # inform MeTA to output log data to stderr
    metapy.log_to_stderr()

    # you will want your version to be >= to this
    metapy.__version__

    print('calling determine_topics() method ...')
    determine_topics('moby_dick-config.toml')

    start_time = time.time()
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))

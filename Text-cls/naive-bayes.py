import json
import nltk
import argparse
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

def preprocess(inputfile, outputfile):
    #TODO: preprocess the input file, and output the result to the output file: train.preprocessed.json,test.preprocessed.json
    #   Delete the useless symbols
    #   Convert all letters to the lowercase
    #   Use NLTK.word_tokenize() to tokenize the sentence
    #   Use nltk.PorterStemmer to stem the words
    nltk.download('punkt')
    stemmer = PorterStemmer()
    translator = str.maketrans('', '', string.punctuation)

    with open(inputfile, 'r') as infile:
        data = json.load(infile)

    preprocessed_data = []

    for record in data:
        file_id = record[0]
        category = record[1]
        text = record[2]
        text = text.translate(translator)
        text = text.lower()
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        preprocessed_data.append({
            'file_id': file_id,
            'category': category,
            'text': ' '.join(stemmed_tokens)
        })
    with open(outputfile, 'w') as outfile:
        json.dump(preprocessed_data, outfile, indent=4)
def count_word(inputfile,outputfile):
    #TODO: count the words from the corpus, and output the result to the output file in the format required.
    #   A dictionary object may help you with this work.
    with open(inputfile, 'r') as infile:
        data = json.load(infile)
    
    word_counts = {}
    class_counts = {'crude': 0, 'grain': 0, 'money-fx': 0, 'acq': 0, 'earn': 0}

    for doc in data:
        category = doc['category']
        class_counts[category] += 1
        words = doc['text'].split()
        for word in words:
            if word not in word_counts:
                word_counts[word] = {'crude': 0, 'grain': 0, 'money-fx': 0, 'acq': 0, 'earn': 0}
            word_counts[word][category] += 1

    with open(outputfile, 'w') as outfile:
        outfile.write(' '.join(str(class_counts[cls]) for cls in ['crude', 'grain', 'money-fx', 'acq', 'earn']) + '\n')
        for word, counts in word_counts.items():
            outfile.write(f"{word} {' '.join(str(counts[cls]) for cls in ['crude', 'grain', 'money-fx', 'acq', 'earn'])}\n")
    return
def feature_selection(inputfile,threshold,outputfile):
    #TODO: Choose the most frequent 10000 words(defined by threshold) as the feature word
    # Use the frequency obtained in 'word_count.txt' to calculate the total word frequency in each class.
    #   Notice that when calculating the word frequency, only words recognized as features are taken into consideration.
    # Output the result to the output file in the format required
    with open(inputfile, 'r') as infile:
        lines = infile.readlines()
    
    class_totals = list(map(int, lines[0].split()))
    word_counts = []

    for line in lines[1:]:
        parts = line.split()
        word = parts[0]
        counts = list(map(int, parts[1:]))
        total_count = sum(counts)
        word_counts.append((word, counts, total_count))
    
    word_counts.sort(key=lambda x: x[2], reverse=True)
    selected_words = word_counts[:threshold]

    with open(outputfile, 'w') as outfile:
        outfile.write(' '.join(map(str, class_totals)) + '\n')
        for word, counts, _ in selected_words:
            outfile.write(f"{word} {' '.join(map(str, counts))}\n")

def calculate_probability(word_count, word_dict, outputfile):
    with open(word_count, 'r') as wc_file, open(word_dict, 'r') as wd_file:
        wc_lines = wc_file.readlines()
        wd_lines = wd_file.readlines()

    class_totals = list(map(int, wd_lines[0].split()))
    total_docs = sum(class_totals)
    prior_probabilities = [count / total_docs for count in class_totals]

    vocabulary = set()
    for line in wd_lines[1:]:
        parts = line.split()
        vocabulary.add(parts[0])
    vocabulary_size = len(vocabulary)

    word_probabilities = {}
    for line in wd_lines[1:]:
        parts = line.split()
        word = parts[0]
        counts = list(map(int, parts[1:]))
        word_probabilities[word] = [
            (count + 1) / (class_totals[i] + vocabulary_size) 
            for i, count in enumerate(counts)
        ]


    with open(outputfile, 'w') as outfile:
        outfile.write(' '.join(map(str, prior_probabilities)) + '\n')
        for word, probabilities in word_probabilities.items():
            outfile.write(f"{word} {' '.join(map(str, probabilities))}\n")

def classify(probability,testset,outputfile):
    #TODO: Implement the naïve Bayes classifier to assign class labels to the documents in the test set.
    #   Output the result to the output file in the format required
    with open(probability, 'r') as prob_file, open(testset, 'r') as test_file:
        prob_lines = prob_file.readlines()
        test_data = json.load(test_file)

    prior_probabilities = list(map(float, prob_lines[0].split()))
    word_probabilities = {}
    for line in prob_lines[1:]:
        parts = line.split()
        word = parts[0]
        probabilities = list(map(float, parts[1:]))
        word_probabilities[word] = probabilities

    results = []
    for doc in test_data:
        file_id = doc['file_id']
        words = doc['text'].split()
        scores = prior_probabilities[:]
        for word in words:
            if word in word_probabilities:
                for i in range(len(scores)):
                    scores[i] *= word_probabilities[word][i]
        predicted_class = ['crude', 'grain', 'money-fx', 'acq', 'earn'][scores.index(max(scores))]
        results.append(f"{file_id} {predicted_class}")

    with open(outputfile, 'w') as outfile:
        outfile.write('\n'.join(results))

def f1_score(testset,classification_result):
    #TODO: Use the F_1 score to assess the performance of the implemented classification model
    #   The return value should be a float object.
    with open(testset, 'r') as test_file, open(classification_result, 'r') as result_file:
        test_data = json.load(test_file)
        results = result_file.readlines()

    true_labels = {doc[0]: doc[1] for doc in test_data}  
    predicted_labels = {line.split()[0]: line.split()[1] for line in results}

    tp = {cls: 0 for cls in ['crude', 'grain', 'money-fx', 'acq', 'earn']}
    fp = {cls: 0 for cls in ['crude', 'grain', 'money-fx', 'acq', 'earn']}
    fn = {cls: 0 for cls in ['crude', 'grain', 'money-fx', 'acq', 'earn']}

    for file_id, true_label in true_labels.items():
        predicted_label = predicted_labels.get(file_id, None)
        if predicted_label == true_label:
            tp[true_label] += 1
        else:
            if predicted_label:
                fp[predicted_label] += 1
            fn[true_label] += 1

    f1_scores = []
    for cls in tp:
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)

def main():
    ''' Main Function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-pps', '--preprocess',type=str,nargs=2,help='preprocess the dataset')
    parser.add_argument('-cw','--count_word',type=str,nargs=2,help='count the words from the corpus')
    parser.add_argument('-fs','--feature_selection',type=str,nargs=3,help='\select the features from the corpus')
    parser.add_argument('-cp','--calculate_probability',type=str,nargs=3,
                        help='calculate the posterior probability of each feature word, and the prior probability of the class')
    parser.add_argument('-cl','--classify',type=str,nargs=3,
                        help='classify the testset documents based on the probability calculated')
    parser.add_argument('-f1','--f1_score', type=str, nargs=2,
                        help='calculate the F-1 score based on the classification result.')
    opt=parser.parse_args()

    if(opt.preprocess):
        input_file = opt.preprocess[0]
        output_file = opt.preprocess[1]
        preprocess(input_file,output_file)
    elif(opt.count_word):
        input_file = opt.count_word[0]
        output_file = opt.count_word[1]
        count_word(input_file,output_file)
    elif(opt.feature_selection):
        input_file = opt.feature_selection[0]
        threshold = int(opt.feature_selection[1])
        outputfile = opt.feature_selection[2]
        feature_selection(input_file,threshold,outputfile)
    elif(opt.calculate_probability):
        word_count = opt.calculate_probability[0]
        word_dict = opt.calculate_probability[1]
        output_file = opt.calculate_probability[2]
        calculate_probability(word_count,word_dict,output_file)
    elif(opt.classify):
        probability = opt.classify[0]
        testset = opt.classify[1]
        outputfile = opt.classify[2]
        classify(probability,testset,outputfile)
    elif(opt.f1_score):
        testset = opt.f1_score[0]
        classification_result = opt.f1_score[1]
        f1 = f1_score(testset,classification_result)
        print('The F1 score of the classification result is: '+str(f1))


if __name__ == '__main__':
    import os
    main()